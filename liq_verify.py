#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lightning_indexer_quant (arch35/950) 算子验证框架: 正确性 + 性能.

背景: test_liq.py 只覆盖 B=1/S1=1/TND+PA/mode=3/topk=2048 的单配置金标准对拍
(其 LIQ_STAGE/LIQ_DUMP 是诊断工具而非断言), 以下维度均未验证:
多token的M块路由 / M与s2尾块 / 多batch混合asl / 非PA key / TND key / BSND无效行 /
topk填充-1 / 零长度批 / q>kv / block_table乱序 / 确定性 / 性能.
本框架按维度系统展开:

正确性 (对拍金标准 = test_liq.py 已验证的打分链路推广到多batch/多布局):
  - 多token: S1=2/4/5/8/17/512 (满M块256、M尾64、gS1多基本块、M轴切换)
  - 多batch与混合asl: 批间不同valid、s2尾块(952/2002/137)、nReal尾块
  - 多核流水: asl_k=8192四块切分+LD归并; B=16单核多基本块(blockS2StartIdx_路径)
  - 布局: query TND/BSND(含无效行), key PA_BSND/BSND/TND, block_table乱序(逻辑!=物理)
  - 掩码: mode=3因果边界(asl_k==asl_q对角线、q>kv、+1语义探针), mode=0
  - topk: 2048/904/8 (CopyOut 裸DataCopy 要求 8 倍数对齐), valid<topk 时 -1 填充
  - 风险探针: topk=900 (非 8 倍数, 契约允许但可能触发 32B 对齐限制, 单列最后)
  - 边界: asl_k=0批(全-1行)、asl_q=0批、BSND缺省asl_q
  - 稳定性: 同输入多次复跑逐位一致
性能:
  - decode/prefill/长序列形状, warmup+多次计时(NPU Event), mean/min/p50 + 吞吐, CSV输出

用法:
  python3.11 liq_verify.py                          # 正确性全量
  python3.11 liq_verify.py --suite perf --iters 10  # 性能
  python3.11 liq_verify.py --suite all
  python3.11 liq_verify.py --list                   # 列出用例
  python3.11 liq_verify.py --filter ld,multiblock   # 名字子串过滤(逗号分隔)
  python3.11 liq_verify.py --seeds 0,1              # 覆盖正确性用例seed
  python3.11 liq_verify.py --csv result.csv

注意:
  - 适配器/tiling 会打印 [LIQ_ADPT]/[LIQ_TILING] 调试行(bring-up脚手架), 本框架输出
    均带 [CASE]/[PERF]/[SUM] 前缀: `python3.11 liq_verify.py 2>/dev/null | grep -E 'CASE|PERF|SUM'`
  - 仅支持 arch35(950) 路径; gSize 固定 64 (N1=64, UB直通端口硬约束)
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

bootstrap_custom_op_env()
from vllm_ascend import vllm_ascend_C  # noqa: F401  注册 torch.ops

DEV = "npu"
BLOCK = 128        # PA 物理块 token 数
G = 64             # N1 = gSize; arch35 UB直通端口硬约束 gSize==64
D = 128            # headDim, tiling 仅支持 128
TOPK_MAX = 2048    # sparse_count 上限
M_BASE = 256       # kernel mBaseSize
S2_BASE = 2048     # kernel s2BaseSize
NEG_INF = float("-inf")


def _norm(v, n, what):
    xs = list(v) if isinstance(v, (list, tuple)) else [v] * n
    assert len(xs) == n, f"{what} 长度 {len(xs)} != B={n}"
    return [int(x) for x in xs]


@dataclass(frozen=True)
class Case:
    name: str
    B: int = 1
    asl_q: tuple = (1,)         # 每batch实际q长度 (标量自动广播)
    asl_k: tuple = (4096,)      # 每batch实际kv长度 (标量自动广播)
    q_layout: str = "TND"
    k_layout: str = "PA_BSND"
    sparse_mode: int = 3        # 3=因果掩码, 0=无掩码
    topk: int = 2048
    q_pad: int = 0              # BSND: qSeqSize 比最大asl_q多出的无效行
    k_pad: int = 0              # s2Size 比最大asl_k多出的余量 (BSND按行; PA按整块)
    shuffle_table: bool = True  # PA: block_table 乱序 (逻辑≠物理)
    omit_asl_q: bool = False    # BSND: 不传 actual_seq_lengths_query (全长度缺省)
    seeds: tuple = (0,)
    repeat: int = 1             # >1 时做逐位一致性复跑
    desc: str = ""


# ============================ 输入构造 ============================

class Inputs:
    pass


def build_inputs(case: Case, seed: int) -> Inputs:
    torch.manual_seed(seed)
    B = case.B
    aq = _norm(case.asl_q, B, "asl_q")
    ak = _norm(case.asl_k, B, "asl_k")
    assert case.q_layout in ("TND", "BSND"), "layout_query 仅支持 TND/BSND"
    assert case.k_layout in ("PA_BSND", "BSND", "TND"), "layout_key 仅支持 PA_BSND/BSND/TND"
    if case.k_layout != "PA_BSND":
        assert case.q_layout == case.k_layout, "非PA key 要求 layout_query == layout_key (tiling约束)"
    assert 0 < case.topk <= TOPK_MAX, f"topk 须在 (0, {TOPK_MAX}]"
    assert all(x >= 0 for x in aq) and all(x >= 0 for x in ak), "asl 不允许负值"

    inp = Inputs()
    inp.case, inp.B, inp.ak, inp.mode, inp.topk = case, B, ak, case.sparse_mode, case.topk

    # ---- query / weights / qScale ----
    if case.q_layout == "TND":
        T = sum(aq)
        assert T > 0, "TND query 需要 sum(asl_q) > 0"
        inp.query = torch.randint(-128, 127, (T, G, D), dtype=torch.int8, device=DEV)
        inp.weights = torch.randn(T, G, dtype=torch.float16, device=DEV)
        inp.q_scale = torch.randn(T, G, dtype=torch.float16, device=DEV)
        # TND: asl_q 为累积长度 (kernel isAccumSeq)
        cum_q = torch.cumsum(torch.tensor(aq, dtype=torch.int64), 0).tolist()
        inp.asl_q_t = torch.tensor(cum_q, dtype=torch.int32, device=DEV)
        inp.rows = [(b, t) for b in range(B) for t in range(aq[b])]
        inp.aq = aq
    else:
        s1max = max(aq) + case.q_pad
        assert s1max > 0, "BSND query 需要 S1max > 0"
        inp.query = torch.randint(-128, 127, (B, s1max, G, D), dtype=torch.int8, device=DEV)
        inp.weights = torch.randn(B, s1max, G, dtype=torch.float16, device=DEV)
        inp.q_scale = torch.randn(B, s1max, G, dtype=torch.float16, device=DEV)
        inp.asl_q_t = None if case.omit_asl_q else torch.tensor(aq, dtype=torch.int32, device=DEV)
        inp.rows = [(b, t) for b in range(B) for t in range(s1max)]
        inp.aq = [s1max] * B if case.omit_asl_q else aq  # 缺省asl_q时按全长度

    # ---- key / kScale / block_table / asl_k ----
    if case.k_layout == "PA_BSND":
        nb = [(k + BLOCK - 1) // BLOCK for k in ak]
        max_nb = max(max(nb), 1)
        pool = B * max_nb
        inp.key = torch.randint(-128, 127, (pool, BLOCK, 1, D), dtype=torch.int8, device=DEV)
        inp.k_scale = torch.randn(pool, BLOCK, 1, dtype=torch.float16, device=DEV)
        bt = torch.empty(B, max_nb, dtype=torch.int32)
        for b in range(B):
            ids = torch.arange(b * max_nb, (b + 1) * max_nb, dtype=torch.int64)
            if case.shuffle_table:
                ids = ids[torch.randperm(max_nb)]
            bt[b] = ids.to(torch.int32)
        inp.block_table = bt.to(DEV)
        inp.asl_k_t = torch.tensor(ak, dtype=torch.int32, device=DEV)  # PA: 非累积
        inp.W = [n * BLOCK for n in nb]  # 每batch逻辑s2宽度
    elif case.k_layout == "BSND":
        s2max = max(max(ak) + case.k_pad, 1)
        inp.key = torch.randint(-128, 127, (B, s2max, 1, D), dtype=torch.int8, device=DEV)
        inp.k_scale = torch.randn(B, s2max, 1, dtype=torch.float16, device=DEV)
        inp.block_table = None
        inp.asl_k_t = torch.tensor(ak, dtype=torch.int32, device=DEV)
        inp.W = [s2max] * B
    else:  # TND key
        T2 = sum(ak)
        assert T2 > 0, "TND key 需要 sum(asl_k) > 0"
        inp.key = torch.randint(-128, 127, (T2, 1, D), dtype=torch.int8, device=DEV)
        inp.k_scale = torch.randn(T2, 1, dtype=torch.float16, device=DEV)
        inp.block_table = None
        cum_k = torch.cumsum(torch.tensor(ak, dtype=torch.int64), 0).tolist()
        inp.asl_k_t = torch.tensor(cum_k, dtype=torch.int32, device=DEV)  # TND key: 累积
        inp.W = list(ak)
    return inp


def run_op(inp: Inputs):
    return torch.ops._C_ascend.npu_lightning_indexer_quant(
        query=inp.query, key=inp.key, weights=inp.weights,
        query_dequant_scale=inp.q_scale, key_dequant_scale=inp.k_scale,
        actual_seq_lengths_query=inp.asl_q_t, actual_seq_lengths_key=inp.asl_k_t,
        block_table=inp.block_table, query_quant_mode=0, key_quant_mode=0,
        layout_query=inp.case.q_layout, layout_key=inp.case.k_layout,
        sparse_count=inp.topk, sparse_mode=inp.mode)


# ============================ 金标准 ============================

@torch.no_grad()
def golden_rows(inp: Inputs, device: str = "cpu") -> List[torch.Tensor]:
    """按输出行序返回每行的 fp32 分数 (无效位置 -inf).

    打分链路 (与 kernel arch35 / test_liq.py 已验证金标准一致):
      dot = fp16(Σ_d q·k / 1024) 再 relu   (DEQF16 + reluPre)
      w'  = fp16(weights · qScale)
      sc  = (Σ_g w'·dot) · fp32(kScale)
    掩码 (mode=3): 行有效长度 = asl_k - asl_q + t + 1 (含对角线, 与v2 arch35一致);
    BSND 无效行 (t >= asl_q[b]) 全 -inf → 期望全 -1.
    """
    B, aq, ak, mode = inp.B, inp.aq, inp.ak, inp.mode
    kl, ql = inp.case.k_layout, inp.case.q_layout
    dev = device
    key, ksc = inp.key.to(dev), inp.k_scale.to(dev)
    qw, qsc, q = inp.weights.to(dev), inp.q_scale.to(dev), inp.query.to(dev)

    if kl == "PA_BSND":
        key_flat = key.view(-1, D)
        ks_flat = ksc.view(-1)
        bt = inp.block_table.to(dev).long()

        def gather_k(b):
            w = inp.W[b]
            if w == 0:
                return None, None
            s = torch.arange(w, device=dev)
            phys = bt[b, s // BLOCK] * BLOCK + (s % BLOCK)
            return key_flat[phys], ks_flat[phys]
    elif kl == "BSND":
        def gather_k(b):
            w = inp.W[b]
            if w == 0:
                return None, None
            return key[b, :w, 0], ksc[b, :w, 0]
    else:  # TND key
        cum_k = torch.cumsum(torch.tensor(ak), 0).tolist()

        def gather_k(b):
            if ak[b] == 0:
                return None, None
            lo = 0 if b == 0 else int(cum_k[b - 1])
            return key[lo:lo + ak[b], 0], ksc[lo:lo + ak[b], 0]

    if ql == "TND":
        cum_q = torch.cumsum(torch.tensor(aq), 0).tolist()

        def gather_q(b):
            lo = 0 if b == 0 else int(cum_q[b - 1])
            return q[lo:lo + aq[b]], qw[lo:lo + aq[b]], qsc[lo:lo + aq[b]], aq[b]
    else:
        def gather_q(b):
            n = inp.query.shape[1]
            return q[b], qw[b], qsc[b], n

    rows = []
    for b in range(B):
        qb, wb, sb, nrows = gather_q(b)
        kt, kst = gather_k(b)
        W = max(inp.W[b], 1)
        if kt is None:
            sc = torch.full((nrows, W), NEG_INF)
        else:
            dot = torch.einsum("tgd,sd->tgs", qb.float(), kt.float()) / 1024.0
            dot = dot.to(torch.float16).float().clamp(min=0)
            w_ = (wb * sb).float()
            sc = torch.einsum("tg,tgs->ts", w_, dot) * kst.float()
            for t in range(nrows):
                if ql == "BSND" and t >= aq[b]:
                    valid = 0
                elif mode == 3:
                    valid = min(max(ak[b] - aq[b] + t + 1, 0), ak[b])
                else:
                    valid = ak[b]
                if valid < W:
                    sc[t, valid:] = NEG_INF
        rows.extend(list(sc))
    assert len(rows) == len(inp.rows), f"金标准行数 {len(rows)} != 期望 {len(inp.rows)}"
    return rows


# ============================ 比对 ============================

def check_outputs(inp: Inputs, out: torch.Tensor, tol_rel: float) -> List[str]:
    fails = []
    out2d = out.reshape(-1, inp.topk).cpu().long()
    golden = golden_rows(inp)
    if out2d.shape[0] != len(golden):
        return [f"输出行数 {out2d.shape[0]} != 期望 {len(golden)} (shape={tuple(out.shape)})"]
    for r, ((b, t), sc) in enumerate(zip(inp.rows, golden)):
        tag = f"行{r}(b{b},t{t})"
        oi = out2d[r]
        valid = int((sc > NEG_INF).sum())
        n = min(valid, inp.topk)
        if n == 0:
            if bool((oi != -1).any()):
                fails.append(f"{tag}: 应全-1, 实际前8={oi[:8].tolist()}")
            continue
        head = oi[:n]
        if int(head.min()) < 0 or int(head.max()) >= valid:
            fails.append(f"{tag}: 索引越界 [{int(head.min())},{int(head.max())}] valid={valid}")
            continue
        if torch.unique(head).numel() != n:
            fails.append(f"{tag}: 索引重复")
            continue
        s_out = sc[head]
        if n > 1 and bool((s_out[1:] > s_out[:-1]).any()):
            fails.append(f"{tag}: 未按分数降序")
            continue
        s_ref = torch.sort(sc, descending=True).values[:n]
        gap = (s_out - s_ref).abs()
        tol = tol_rel * s_ref.abs().clamp(min=1.0)
        bad = int((gap > tol).sum())
        if bad:
            i = int(gap.argmax())
            fails.append(f"{tag}: {bad}/{n} 名次分数超差 (最大 {float(gap[i]):.4f} @名次{i}: "
                         f"出 {float(s_out[i]):.3f} vs 参 {float(s_ref[i]):.3f})")
            continue
        if n < inp.topk and not bool((oi[n:] == -1).all()):
            fails.append(f"{tag}: 尾部未填-1 (前8={oi[n:n + 8].tolist()})")
    return fails


# ============================ 用例表 ============================

CORRECTNESS_CASES = [
    Case("base_tnd_pa", seeds=(0, 1, 2), desc="T1基线回归: 单token/PA/TND/mode3, 2核切分+LD"),
    Case("s1_4_full_m", asl_q=(4,), seeds=(0, 1, 2), desc="满M块256(2个L0块): dualDst双token路由"),
    Case("s1_5_m_tail", asl_q=(5,), seeds=(0, 1), desc="M尾块64: 末L0块半块, 无效token跳过"),
    Case("mask_diag_probe", asl_q=(2,), asl_k=(2,), topk=8,
         desc="因果+1语义探针: token0恰好看pos0 (抓off-by-one)"),
    Case("mask_diag_s1_8", asl_q=(8,), asl_k=(8,), topk=8, desc="asl_k==asl_q 全对角"),
    Case("no_mask_mode0", asl_q=(2,), asl_k=(300,), sparse_mode=0, topk=128, desc="mode=0无掩码"),
    Case("b4_mixed_asl", B=4, asl_q=(1, 4, 1, 2), asl_k=(4096, 3000, 2048, 137), seeds=(0, 1),
         desc="多batch混合asl: s2尾块952/137, nReal尾块"),
    Case("ld_split_8k", asl_k=(8192,), desc="4个s2块切4核: LD归并4路partial"),
    Case("multiblock_per_core", B=16, asl_q=(1,), asl_k=(4096,),
         desc="32块/24核: 单核多基本块(blockS2StartIdx_+头尾ws)"),
    Case("gs1_multi", asl_q=(17,), desc="5个gS1块(4满+尾64)x2 s2块: M轴切换重初始化"),
    Case("topk_pad_904", asl_k=(700,), topk=904, desc="valid<topk → -1填充; sparseCount<2048"),
    Case("topk_8", asl_k=(2048,), topk=8, desc="topk最小对齐值(32B边界)"),
    Case("bsnd_q_pad", B=2, asl_q=(3, 5), q_pad=3, asl_k=(2048, 4096),
         desc="BSND query + 无效行-1 + 批间不同asl"),
    Case("key_bsnd", q_layout="BSND", k_layout="BSND", B=2, asl_q=(2, 2), asl_k=(2048, 3000),
         desc="非PA key (BSND, 无block_table)"),
    Case("key_tnd", q_layout="TND", k_layout="TND", B=2, asl_q=(2, 2), asl_k=(2048, 3000),
         desc="TND key 打包 + 累积asl_k"),
    Case("zero_len_batches", B=4, asl_q=(1, 2, 0, 1), asl_k=(2048, 0, 1024, 512),
         desc="asl_k=0批(全-1行) + asl_q=0批"),
    Case("mask_q_gt_kv", asl_q=(4,), asl_k=(2,), topk=8, desc="q长于kv: 前行全-1(invalidS1Num2)"),
    Case("pa_identity_table", shuffle_table=False, desc="block_table恒等映射(对照组)"),
    Case("omit_asl_q_bsnd", q_layout="BSND", asl_q=(4,), omit_asl_q=True, asl_k=(2048,),
         desc="BSND不传asl_q(全长度缺省路径)"),
    Case("stability_repeat", asl_q=(4,), repeat=5, desc="同输入5次逐位一致(确定性)"),
    # 已知风险探针: CopyOut为裸DataCopy, topk非8倍数时长度非32B对齐, op契约又允许任意
    # sparse_count(0,2048] —— 该用例可能失败甚至设备挂死, 排在最后, 可--filter排除
    Case("topk_align_probe", asl_k=(700,), topk=900,
         desc="风险探针: topk=900非8倍数, CopyOut 32B对齐限制"),
]

PERF_CASES = [
    Case("perf_small", desc="基线: B1/t1/kv4k/topk2048"),
    Case("perf_decode_b32", B=32, asl_k=(8192,), desc="decode: B32/t1/kv8k"),
    Case("perf_prefill", B=4, asl_q=(512,), asl_k=(16384,), desc="prefill: B4x512tok/kv16k"),
    Case("perf_longctx", B=2, asl_q=(1024,), asl_k=(65536,),
         desc="长序列: B2x1024tok/kv64k (耗时大, 可--filter排除)"),
]


# ============================ 执行器 ============================

def run_correctness(case: Case, seeds, tol_rel):
    details = []
    ok = True
    for seed in seeds:
        try:
            inp = build_inputs(case, seed)
            outs = []
            for _ in range(case.repeat):
                outs.append(run_op(inp).cpu())
                torch.npu.synchronize()  # aicore 异步错误在此归位到当前用例
            fails = check_outputs(inp, outs[0], tol_rel)
            for i in range(1, case.repeat):
                if not torch.equal(outs[i], outs[0]):
                    fails.append(f"第{i}次复跑与首次不一致 (非确定性)")
            if fails:
                ok = False
                show = "; ".join(fails[:3]) + (" ..." if len(fails) > 3 else "")
                details.append(f"seed{seed}: {show}")
        except Exception as ex:  # noqa: BLE001  用例级兜底: aicore错误/断言都算FAIL
            ok = False
            details.append(f"seed{seed}: 异常 {type(ex).__name__}: {str(ex)[:180]}")
    return ok, details


def est_blocks(inp: Inputs) -> int:
    """估算kernel基本块数 (与kernel SplitCore同公式), 仅用于吞吐报告."""
    s1_base = M_BASE // G
    total = 0
    for b in range(inp.B):
        s1g = (inp.aq[b] * G + M_BASE - 1) // M_BASE
        if inp.mode == 3:
            for i in range(s1g):
                v = min(max(s1_base * i + (inp.ak[b] - inp.aq[b]) + s1_base, 1), inp.ak[b])
                total += (v + S2_BASE - 1) // S2_BASE
        else:
            total += s1g * ((inp.ak[b] + S2_BASE - 1) // S2_BASE)
    return total


def run_perf(case: Case, warmup: int, iters: int):
    inp = build_inputs(case, 0)
    for _ in range(warmup):
        run_op(inp)
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        s = torch.npu.Event(enable_timing=True)
        e = torch.npu.Event(enable_timing=True)
        torch.npu.synchronize()
        s.record()
        run_op(inp)
        e.record()
        torch.npu.synchronize()
        times.append(s.elapsed_time(e))  # ms
    times.sort()
    mean = sum(times) / len(times)
    qtok = sum(inp.aq)
    pairs = sum(q * k for q, k in zip(inp.aq, inp.ak))
    blocks = est_blocks(inp)
    return {
        "iters": iters, "mean_ms": mean, "min_ms": times[0], "p50_ms": times[len(times) // 2],
        "qtok": qtok, "pairs": pairs, "blocks": blocks,
        "qtok_s": qtok * 1000.0 / mean, "pairs_s": pairs * 1000.0 / mean,
        "blocks_s": blocks * 1000.0 / mean,
    }


# ============================ main ============================

def main():
    ap = argparse.ArgumentParser(description="lightning_indexer_quant 验证框架")
    ap.add_argument("--suite", choices=["correctness", "perf", "all"], default="correctness")
    ap.add_argument("--filter", default="", help="用例名子串过滤, 逗号分隔")
    ap.add_argument("--seeds", default="", help="覆盖正确性用例seed, 逗号分隔 (如 0,1,2)")
    ap.add_argument("--tol", type=float, default=1e-3, help="名次分数相对容差")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--csv", default="", help="结果CSV输出路径")
    ap.add_argument("--list", action="store_true", help="仅列出用例")
    args = ap.parse_args()

    if not torch.npu.is_available():
        sys.exit("[SUM] 错误: 无可用NPU设备")

    def selected(cases):
        if not args.filter:
            return cases
        pats = [p.strip().lower() for p in args.filter.split(",") if p.strip()]
        return [c for c in cases if any(p in c.name.lower() for p in pats)]

    corr = selected(CORRECTNESS_CASES) if args.suite in ("correctness", "all") else []
    perf = selected(PERF_CASES) if args.suite in ("perf", "all") else []

    if args.list:
        for c in CORRECTNESS_CASES + PERF_CASES:
            print(f"{c.name:24s} B={c.B} asl_q={c.asl_q} asl_k={c.asl_k} "
                  f"{c.q_layout}/{c.k_layout} mode={c.sparse_mode} topk={c.topk} # {c.desc}")
        return

    seeds_override = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "") if args.seeds else None
    rows_csv = []
    n_pass = n_fail = 0

    for c in corr:
        seeds = seeds_override if seeds_override else c.seeds
        ok, details = run_correctness(c, seeds, args.tol)
        status = "PASS" if ok else "FAIL"
        n_pass += ok
        n_fail += not ok
        print(f"[CASE] {status} {c.name:24s} seeds={len(seeds)} repeat={c.repeat} # {c.desc}")
        for d in details:
            print(f"[CASE]    -> {d}")
        rows_csv.append({"suite": "correctness", "name": c.name, "status": status,
                         "detail": " | ".join(details)[:500]})

    for c in perf:
        try:
            m = run_perf(c, args.warmup, args.iters)
            print(f"[PERF] {c.name:24s} mean={m['mean_ms']:.3f}ms p50={m['p50_ms']:.3f}ms "
                  f"min={m['min_ms']:.3f}ms | qtok/s={m['qtok_s']:.0f} "
                  f"pairs/s={m['pairs_s']:.3g} blocks/s={m['blocks_s']:.1f} # {c.desc}")
            rows_csv.append({"suite": "perf", "name": c.name, "status": "OK", "detail": "",
                             **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        except Exception as ex:  # noqa: BLE001
            print(f"[PERF] ERROR {c.name:24s} {type(ex).__name__}: {str(ex)[:180]} # {c.desc}")
            rows_csv.append({"suite": "perf", "name": c.name, "status": "ERROR",
                             "detail": str(ex)[:500]})

    print(f"[SUM] 正确性: {n_pass}/{n_pass + n_fail} PASS; 性能: {len(perf)} 个形状")
    if n_fail:
        names = [r["name"] for r in rows_csv if r["suite"] == "correctness" and r["status"] == "FAIL"]
        print(f"[SUM] 失败用例: {', '.join(names)}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["suite", "name", "status", "detail"] +
                               ["iters", "mean_ms", "min_ms", "p50_ms", "qtok", "pairs", "blocks",
                                "qtok_s", "pairs_s", "blocks_s"])
            w.writeheader()
            w.writerows(rows_csv)
        print(f"[SUM] CSV 已写入 {args.csv}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
