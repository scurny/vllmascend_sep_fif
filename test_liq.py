import os

import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

# 裸导入不走 vLLM 引擎初始化, 必须手动把 vllm_ascend/_cann_ops_custom
# 前置到 ASCEND_CUSTOM_OPP_PATH, 否则会加载环境变量里残留的旧算子包
bootstrap_custom_op_env()
from vllm_ascend import vllm_ascend_C  # noqa: F401  注册 torch.ops

torch.manual_seed(0)

# ---------------- 公共输入 (贴近生产: 有效 KV 长度 > topk) ----------------
# 形状可用环境变量缩小, 用于排查"崩点是否与计算规模相关" (如 LIQ_TOPK=64 LIQ_ASL=128 LIQ_BLOCKNUM=2)
B, S1, N1, D = 1, 1, 64, 128
block_num = int(os.environ.get("LIQ_BLOCKNUM", 100))
block_size = 128
asl_k = int(os.environ.get("LIQ_ASL", 4096))   # 有效 KV token 数, 须 <= block_num*block_size 且 > topk
topk = int(os.environ.get("LIQ_TOPK", 2048))

def build_inputs(seed=0):
    """构建一组输入(全局变量), 便于多 seed 对拍"""
    global query, key, weights, query_dequant_scale, key_dequant_scale, \
        actual_seq_lengths_query, actual_seq_lengths_key, block_table
    torch.manual_seed(seed)
    query = torch.randint(-128, 127, (B * S1, N1, D), dtype=torch.int8, device='npu')
    key = torch.randint(-128, 127, (block_num, block_size, 1, D), dtype=torch.int8, device='npu')
    weights = torch.randn(B * S1, N1, dtype=torch.float16, device='npu')
    query_dequant_scale = torch.randn(B * S1, N1, dtype=torch.float16, device='npu')
    key_dequant_scale = torch.randn(block_num, block_size, 1, dtype=torch.float16, device='npu')
    actual_seq_lengths_query = torch.tensor([S1], dtype=torch.int32, device='npu')
    actual_seq_lengths_key = torch.tensor([asl_k], dtype=torch.int32, device='npu')
    block_table = torch.arange(block_num, dtype=torch.int32, device='npu').view(B, -1)


build_inputs(0)


def run_op(**kw):
    return torch.ops._C_ascend.npu_lightning_indexer_quant(
        query=query, key=key, weights=weights,
        query_dequant_scale=query_dequant_scale,
        key_dequant_scale=key_dequant_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_key=actual_seq_lengths_key,
        block_table=block_table,
        query_quant_mode=0, key_quant_mode=0,
        layout_query="TND", layout_key="PA_BSND",
        sparse_count=topk, sparse_mode=3, **kw)


def check_out(tag, out):
    torch.npu.synchronize()  # aicore 异步错误在此暴露, 而不是退出时
    print(tag, "shape:", tuple(out.shape), "dtype:", out.dtype)
    print(tag, "sample:", out.flatten()[:16].cpu().tolist())
    print(tag, "min/max:", out.min().item(), out.max().item())  # 索引应在 [-1, asl_k) 内
    if os.environ.get("LIQ_FILL"):
        of = out.flatten().cpu()
        sent_mask = of == 0x7E7E7E7E
        n_sent = int(sent_mask.sum())
        non_sent = of[~sent_mask]
        n_valid = int(((non_sent >= 0) & (non_sent < asl_k)).sum()) if non_sent.numel() else 0
        pos = (~sent_mask).nonzero().flatten().tolist()
        vals = non_sent.tolist()
        asc = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
        print(tag, f"[FILL] kernel 覆盖 {of.numel() - n_sent}/{of.numel()} 单元, "
                   f"非哨兵中合法索引数 {n_valid}; 非哨兵前12: {vals[:12]}")
        print(tag, f"[FILL] 写入位置 首10={pos[:10]} 尾10={pos[-10:]}; "
                   f"值升序={asc}, 唯一数={len(set(vals))}, 后12={vals[-12:]}")
        if vals:
            vmin, vmax = min(vals), max(vals)
            nb = 16
            hist = [0] * nb
            for v in vals:
                hist[min(nb - 1, v * nb // max(asl_k, 1))] += 1
            wraps = [i for i in range(1, len(vals)) if vals[i] < vals[i - 1]]
            winfo = [(vals[w - 1], vals[w]) for w in wraps[:5]]
            print(tag, f"[FILL] 值域 [{vmin},{vmax}], 直方图16桶={hist}, 前40={vals[:40]}")
            print(tag, f"[FILL] 折返点数={len(wraps)}, 前5折返处(前值->后值)={winfo}")


# ---------------- 金标准参考 (与 kernel arch35 精度模型逐位对齐) ----------------
# kernel 打分链路 (见 service_cube/vector_arch35.h):
#   C_qk[s1*g, s2] = Σ_d q[s1,g,d]·k[s2,d]           (int8 精确累加, int32)
#   S = fp16(C_qk · 2^-10)                            (Fixpipe DEQF16, prequant 0x3a800000)
#   w' = fp16(weights·qScale)                         (Vec0 half 乘法)
#   C_ws[s1, s2] = Σ_g w'[s1,g]·S[s1*g+g, s2]         (Mmad half×half → fp32 累加)
#   score[s1, s2] = C_ws[s1, s2] · fp32(kScale[s2])   (Vec1)
# sparse_mode=3 (causal 下三角): 第 t 行有效长度 = asl_k - asl_q + 1 + t
def liq_golden():
    q = query.cpu().float()                                    # [T, G, D]
    blk, bs = key.shape[0], key.shape[1]
    bt = block_table[0].cpu().long()
    s2 = torch.arange(blk * bs)
    phys = bt[s2 // bs] * bs + (s2 % bs)                       # 逻辑 token -> 物理行 (PA 布局)
    k_tokens = key.view(blk * bs, -1).cpu().float()[phys]      # [S2, D]
    dot = torch.einsum('tgd,sd->tgs', q, k_tokens) / 1024.0    # int8 点积, fp32 精确承载
    dot = dot.to(torch.float16).float().clamp(min=0)           # DEQF16 舍入 + reluPre (FixpSToL1 reluPre=1)
    w = (weights.cpu() * query_dequant_scale.cpu()).float()    # fp16 乘法舍入后转 fp32
    score = torch.einsum('tg,tgs->ts', w, dot)                 # fp32 头累加
    ks = key_dequant_scale.view(blk * bs).cpu().float()[phys]  # [S2]
    score = score * ks
    t = torch.arange(q.shape[0])
    valid = (asl_k - S1 + 1 + t).clamp(0, asl_k)               # causal 有效长度
    for r in range(q.shape[0]):
        score[r, valid[r]:] = float('-inf')
    return score


def formula_sweep(tag, written_set):
    """候选公式扫描: kernel 写入集合与哪种打分公式的 top-k 吻合, 定位打分链路偏差.
    证据: ema=200000 触发 3 token (~原始点积 3.3σ), 金标准 std 仅 ~472 (423σ 不可能触发)."""
    q = query.cpu().float()
    blk, bs = key.shape[0], key.shape[1]
    bt = block_table[0].cpu().long()
    s2 = torch.arange(blk * bs)
    phys = bt[s2 // bs] * bs + (s2 % bs)
    k_tokens = key.view(blk * bs, -1).cpu().float()[phys]
    dot = torch.einsum('tgd,sd->tgs', q, k_tokens)            # 原始 int8 点积, std ~6e4
    dot_fp16 = dot.to(torch.float16).float()                   # fp16 饱和截断 (无 /1024 时)
    w = (weights.cpu() * query_dequant_scale.cpu()).float()
    w_raw = weights.cpu().float()
    ks = key_dequant_scale.view(blk * bs).cpu().float()[phys]
    cands = {
        'dot_g0': dot[0, 0],                                   # 单头原始点积
        'dot_glast': dot[0, -1],
        'dot_sum': dot.sum(1)[0],                              # 头求和 (无权重)
        'dot_sum_x_ks': (dot.sum(1)[0]) * ks,
        'no_deq_sat': ((w.unsqueeze(2) * dot_fp16).sum(1)[0]) * ks,   # 缺 /1024, fp16 饱和
        'w_only_no_qs': ((w_raw.unsqueeze(2) * (dot / 1024.0).to(torch.float16).float()).sum(1)[0]) * ks,
        'ks_only': ks.clone(),
    }
    k_top = min(topk, asl_k)
    parts = []
    for name, sc in cands.items():
        sc = sc.clone()
        sc[asl_k:] = float('-inf')
        top = set(torch.topk(sc, k_top).indices.tolist())
        parts.append(f"{name}={len(top & written_set)}")
    print(f"{tag} [SWEEP] 候选公式 top-k 与写入集合交集: {', '.join(parts)} (金标准基线见[SET])")


FAILURES = []


def check_topk_vs_golden(tag, out, score):
    """逐名次比对: 索引不重复/不越界/按分数降序/名次分数与参考一致(容忍 fp16 平分扰动)"""
    if os.environ.get("LIQ_FILL"):
        of = out.flatten().cpu()
        body = of[1:]  # 跳过 out[0] (marker 写入位)
        written = body[body != 0x7E7E7E7E].long().tolist()
        valid_len = int((score[0] > float('-inf')).sum())
        ref = torch.argsort(-score[0])[:min(valid_len, topk)].tolist()
        inter = len(set(written) & set(ref))
        print(f"{tag} [SET] kernel 写入 {len(written)} 个值 (唯一 {len(set(written))}), "
              f"参考 top{len(ref)} 集合交集 {inter}")
        formula_sweep(tag, set(written))
    out_idx = out.reshape(-1, out.shape[-1]).cpu()
    for r in range(out_idx.shape[0]):
        oi = out_idx[r].long()
        valid_len = int((score[r] > float('-inf')).sum())
        cmp_len = min(valid_len, topk)
        if cmp_len > 0 and (int(oi[:cmp_len].min()) < 0 or int(oi[:cmp_len].max()) >= valid_len):
            FAILURES.append(f"{tag} row{r}: 索引越界 (min={int(oi.min())}, max={int(oi.max())}, valid={valid_len})")
            return
        if torch.unique(oi[:cmp_len]).numel() != cmp_len:
            FAILURES.append(f"{tag} row{r}: 存在重复索引")
            return
        s_out = score[r, oi[:cmp_len]]
        if cmp_len > 1 and bool((s_out[1:] > s_out[:-1]).any()):
            FAILURES.append(f"{tag} row{r}: 输出未按分数降序")
            return
        s_ref = torch.sort(score[r], descending=True).values[:cmp_len]
        gap = (s_out - s_ref).abs()
        tol = 1e-3 * s_ref.abs().clamp(min=1.0)
        nbad = int((gap > tol).sum())
        if nbad:
            FAILURES.append(f"{tag} row{r}: {nbad}/{cmp_len} 个名次分数与参考偏差超容忍 "
                            f"(最大偏差 {float(gap.max()):.4f})")
            return
        if cmp_len < topk and not torch.equal(oi[cmp_len:], torch.full_like(oi[cmp_len:], -1)):
            FAILURES.append(f"{tag} row{r}: 有效长度不足 topk 时尾部应填充 -1")
            return
    top = out_idx[0][:8].tolist()
    tops = ", ".join(f"{i}:{score[0, i]:.1f}" for i in top)
    print(f"{tag} 金标准对拍通过 (top8: {tops})")


# ---------------- 崩溃二分: LIQ_STAGE=1/2/3 时只跑对应阶段 gate (ema_threshold 传负值) ----------------
import os
import sys

_stage = os.environ.get("LIQ_STAGE")
if _stage:
    _s = float(_stage)
    _with_pf = os.environ.get("LIQ_WITH_PF") == "1"
    print(f"=== STAGE {_s:g}: kernel gate ema_threshold={-_s:g} with_pf={_with_pf} ===")
    _kw = {}
    if _with_pf:
        # 带 prefetch 张量: 让 kernel 参数门(依赖 prefetchStagingBuf != null)不触发,
        # 从而走到 tiling 读取 + 标记写入 + ema 门
        _kseq = block_num * block_size
        _tsbk, _keb = 256, 384
        _kp = torch.randint(0, 256, (_kseq, _tsbk), dtype=torch.uint8, device='npu')
        _vp = torch.randint(0, 256, (_kseq, _keb - _tsbk), dtype=torch.uint8, device='npu')
        _ar = torch.arange(_kseq)
        _gva = torch.empty((B, _kseq, 2), dtype=torch.int64)
        _gva[:, :, 0] = _kp.data_ptr() + _ar * _tsbk
        _gva[:, :, 1] = _vp.data_ptr() + _ar * (_keb - _tsbk)
        _kw = dict(token_gva_table=_gva.to('npu'),
                   prefetch_staging_buf=torch.zeros((topk * 2, _keb), dtype=torch.uint8, device='npu'),
                   prefetch_slot_counter=torch.zeros(1, dtype=torch.int32, device='npu'),
                   token_to_prefetch_slot=torch.full((B, _kseq), -1, dtype=torch.int32, device='npu'),
                   token_size_bytes_k=_tsbk)
    _out = None
    try:
        _out = run_op(ema_threshold=-_s, **_kw)
        torch.npu.synchronize()
        print(f"=== STAGE {_s:g} OK (无 aicore 错误) ===")
    except RuntimeError as e:
        print(f"=== STAGE {_s:g} aicore 异常: {str(e)[:200]} ===")
    if _out is not None:
        # kernel 入口调试写入: out[0] = kernel 侧读到的 emaThreshold 位型
        try:
            _raw = int(_out.flatten()[0].item()) & 0xFFFFFFFF
            _exp = {1: 0xBF800000, 2: 0xC0000000, 3: 0xC0400000, 14: 0xC1600000, 15: 0xC1700000,
                    11: 0xC1300000, 12: 0xC1400000, 13: 0xC1500000}.get(int(_s), 0xBF800000)
            if _raw == _exp:
                _hint = f"等于期望的 -{_s:g}f 位型, 阶段门在跑"
            elif _raw == 0x7E7E7E7E:
                _hint = ("参数门生效(kernel 未读 tiling 即返回, 预期结果)" if not _with_pf
                         else "哨兵未覆盖 -> kernel 没执行入口写入, 旧二进制!")
            elif _raw == 0x00000000:
                _hint = "kernel 读到 0.0 -> emaThreshold 字段没进 tiling buffer(截断/未初始化), 对照 [LIQ_TILING] TilingBuf 的 capacity 和 structSize"
            else:
                _hint = "其他值 -> kernel 读到的 emaThreshold 不是预期值, tiling 数据传递有问题"
            print(f"[LIQ_DEBUG] out[0] = 0x{_raw:08X} (期望 0x{_exp:08X}): {_hint}")
        except Exception as e2:
            print(f"[LIQ_DEBUG] 输出读取失败(设备异常态): {str(e2)[:120]}")
    sys.exit(0)

# ---------------- mm1Res 落盘诊断: ema=-20, Vec1 把 mm1Res 原始位型写入输出 ----------------
# 判读: 每个输出单元是 mm1Res 原始 float 位型; 按 128-token tile 与金标准 C_ws 比对
# (直方图实测: 偶数 tile(>=2) 的 token 从未被选中 -> 疑似 mm1Res 相应区段未被 cube 落盘)
if os.environ.get("LIQ_DUMP"):
    build_inputs(0)
    out = run_op(ema_threshold=-20.0)
    torch.npu.synchronize()
    raw = out.flatten().cpu().to(torch.int32)
    floats = raw.view(torch.float32)
    q = query.cpu().float()
    blk, bs = key.shape[0], key.shape[1]
    bt = block_table[0].cpu().long()
    s2 = torch.arange(blk * bs)
    phys = bt[s2 // bs] * bs + (s2 % bs)
    k_tokens = key.view(blk * bs, -1).cpu().float()[phys]
    dot = torch.einsum('tgd,sd->tgs', q, k_tokens) / 1024.0
    dot = dot.to(torch.float16).float().clamp(min=0)
    w = (weights.cpu() * query_dequant_scale.cpu()).float()
    cws = torch.einsum('tg,tgs->ts', w, dot)[0]
    tile = 128
    ntile = asl_k // tile
    n_sent = int((raw == 0x7E7E7E7E).sum())
    print(f"=== LIQ_DUMP: 共 {raw.numel()} 单元, 哨兵(未写入) {n_sent}, "
          f"前8位型: {[hex(x & 0xFFFFFFFF) for x in raw[:8].tolist()]}")
    for t in range(ntile):
        seg_i = raw[t * tile:(t + 1) * tile]
        seg = floats[t * tile:(t + 1) * tile]
        ref = cws[t * tile:(t + 1) * tile]
        if int((seg_i == 0x7E7E7E7E).sum()) == tile:
            verdict = "未写入(哨兵)"
        else:
            diff = float((seg - ref).abs().max())
            verdict = "OK" if diff <= 0.05 * float(ref.abs().max()) + 2.0 else "BAD"
        print(f"[DUMP] tile {t:2d} [{t*tile:5d},{(t+1)*tile:5d}): {verdict:12s} "
              f"kernel前4={seg[:4].tolist()} 金标前4={[round(x,1) for x in ref[:4].tolist()]}")
    sys.exit(0)


# ---------------- Test 1: 不带 prefetch, 多 seed 金标准对拍 ----------------
for seed in (0, 1, 2):
    build_inputs(seed)
    out = run_op()
    check_out(f"[T1 s{seed}]", out)
    _n = len(FAILURES)
    check_topk_vs_golden(f"[T1 seed={seed}]", out, liq_golden())
    print(f"=== Test 1 seed={seed} {'FAIL' if len(FAILURES) > _n else 'PASS'} ===")
build_inputs(0)  # 恢复 seed 0, 供后续测试与金标准

# ---------------- prefetch 公共张量 ----------------
# kernel 语义: s2Size = block_num * block_size (PA 布局),
# token_gva_table 按 [B, s2Size, 2] 索引 (K 地址, V 地址),
# token_to_prefetch_slot 按 [B, s2Size] 索引, 均须覆盖全部 token
token_size_bytes_k = 256
k_entry_bytes = 384
prefetch_candidate_max = topk * 2
kSeqSize = block_num * block_size

k_pool = torch.randint(0, 256, (kSeqSize, token_size_bytes_k), dtype=torch.uint8, device='npu')
v_pool = torch.randint(0, 256, (kSeqSize, k_entry_bytes - token_size_bytes_k), dtype=torch.uint8, device='npu')
ar = torch.arange(kSeqSize)
gva = torch.empty((B, kSeqSize, 2), dtype=torch.int64)
gva[:, :, 0] = k_pool.data_ptr() + ar * token_size_bytes_k
gva[:, :, 1] = v_pool.data_ptr() + ar * (k_entry_bytes - token_size_bytes_k)
token_gva_table = gva.to('npu')


def make_prefetch_tensors():
    return (torch.zeros((prefetch_candidate_max, k_entry_bytes), dtype=torch.uint8, device='npu'),
            torch.zeros(1, dtype=torch.int32, device='npu'),
            torch.full((B, kSeqSize), -1, dtype=torch.int32, device='npu'))


# ---------------- Test 2a: prefetch 开启但零触发 ----------------
print("=== Test 2a: prefetch 开启, 阈值巨大 (零触发) ===")
_n = len(FAILURES)
staging, counter, slot_map = make_prefetch_tensors()
out2a = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=1e30, token_size_bytes_k=token_size_bytes_k)
check_out("[T2a]", out2a)
check_topk_vs_golden("[T2a]", out2a, liq_golden())  # prefetch 不应影响主输出
used2a = int((slot_map != -1).sum().item())
print("[T2a] used slots:", used2a, "(应为 0)")
if used2a != 0:
    FAILURES.append(f"[T2a] 零触发阈值下仍占用 {used2a} 个 slot")
print(f"=== Test 2a {'FAIL' if len(FAILURES) > _n else 'PASS'} ===")

# ---------------- Test 2b: 真实 prefetch ----------------
# 阈值调参: 随机数据下 score 标准差约 ~6e4, 200000 约 3σ 只触发少量 token。
# kernel 内 bitPositions 数组上限 256: 单 tile 超阈值 token 数须远小于 256,
# used=0 则降到 100000, 崩了则升到 1000000
print("=== Test 2b: 真实 prefetch ===")
_n = len(FAILURES)
staging, counter, slot_map = make_prefetch_tensors()
out2b = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=200000.0, token_size_bytes_k=token_size_bytes_k)
check_out("[T2b]", out2b)
check_topk_vs_golden("[T2b]", out2b, liq_golden())  # prefetch 不应影响主输出
used2b = int((slot_map[0] != -1).sum().item())
print("[T2b] used slots:", used2b)
if used2b == 0:
    FAILURES.append("[T2b] ema=200000 未触发任何 prefetch, 无法验证搬运正确性 (考虑降低阈值)")
else:
    bad = 0
    for tok in (slot_map[0] != -1).nonzero().flatten()[:8].tolist():
        slot = int(slot_map[0, tok].item())
        k_ok = torch.equal(staging[slot, :token_size_bytes_k].cpu(), k_pool[tok].cpu())
        v_ok = torch.equal(staging[slot, token_size_bytes_k:].cpu(), v_pool[tok].cpu())
        if not (k_ok and v_ok):
            bad += 1
            row = staging[slot].cpu()
            print(f"[T2b] token {tok} -> slot {slot}: K={k_ok} V={v_ok}, staging全零={bool((row == 0).all())}, "
                  f"staging前8B={row[:8].tolist()}, 期望前8B={k_pool[tok][:8].cpu().tolist()}")
    if bad:
        FAILURES.append(f"[T2b] {bad} 个 slot 搬运内容与源不符")
print(f"=== Test 2b {'FAIL' if len(FAILURES) > _n else 'PASS'} ===")

if FAILURES:
    print("==== 存在失败项 ====")
    for f in FAILURES:
        print(" -", f)
    sys.exit(1)
print("==== ALL PASS ====")
