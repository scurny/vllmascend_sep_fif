import torch
import torch_npu

from vllm_ascend.utils import bootstrap_custom_op_env

# 裸导入不走 vLLM 引擎初始化, 必须手动把 vllm_ascend/_cann_ops_custom
# 前置到 ASCEND_CUSTOM_OPP_PATH, 否则会加载环境变量里残留的旧算子包
bootstrap_custom_op_env()
from vllm_ascend import vllm_ascend_C  # noqa: F401  注册 torch.ops

torch.manual_seed(0)

# ---------------- 公共输入 (贴近生产: 有效 KV 长度 > topk) ----------------
B, S1, N1, D = 1, 1, 64, 128
block_num, block_size = 100, 128
asl_k = 4096            # 有效 KV token 数, 须 <= block_num*block_size=12800 且 > topk
topk = 2048

query = torch.randint(-128, 127, (B * S1, N1, D), dtype=torch.int8, device='npu')
key = torch.randint(-128, 127, (block_num, block_size, 1, D), dtype=torch.int8, device='npu')
weights = torch.randn(B * S1, N1, dtype=torch.float16, device='npu')
query_dequant_scale = torch.randn(B * S1, N1, dtype=torch.float16, device='npu')
key_dequant_scale = torch.randn(block_num, block_size, 1, dtype=torch.float16, device='npu')
actual_seq_lengths_query = torch.tensor([S1], dtype=torch.int32, device='npu')
actual_seq_lengths_key = torch.tensor([asl_k], dtype=torch.int32, device='npu')
block_table = torch.arange(block_num, dtype=torch.int32, device='npu').view(B, -1)


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


# ---------------- 崩溃二分: LIQ_STAGE=1/2/3 时只跑对应阶段 gate (ema_threshold 传负值) ----------------
import os
import sys

_stage = os.environ.get("LIQ_STAGE")
if _stage:
    _s = float(_stage)
    print(f"=== STAGE {_s:g}: kernel gate ema_threshold={-_s:g} ===")
    _out = None
    try:
        _out = run_op(ema_threshold=-_s)
        torch.npu.synchronize()
        print(f"=== STAGE {_s:g} OK (无 aicore 错误) ===")
    except RuntimeError as e:
        print(f"=== STAGE {_s:g} aicore 异常: {str(e)[:200]} ===")
    if _out is not None:
        # kernel 入口调试写入: out[0] = kernel 侧读到的 emaThreshold 位型
        try:
            _raw = int(_out.flatten()[0].item()) & 0xFFFFFFFF
            _exp = {1: 0xBF800000, 2: 0xC0000000, 3: 0xC0400000}.get(int(_s), 0xBF800000)
            if _raw == _exp:
                _hint = f"等于期望的 -{_s:g}f 位型, 阶段门在跑"
            elif _raw == 0x7E7E7E7E:
                _hint = "哨兵未覆盖 -> kernel 根本没执行入口写入, 跑的是不含调试补丁的旧二进制!"
            else:
                _hint = "其他值 -> kernel 读到的 emaThreshold 不是预期值, tiling 数据传递有问题"
            print(f"[LIQ_DEBUG] out[0] = 0x{_raw:08X} (期望 0x{_exp:08X}): {_hint}")
        except Exception as e2:
            print(f"[LIQ_DEBUG] 输出读取失败(设备异常态): {str(e2)[:120]}")
    sys.exit(0)

# ---------------- Test 1: 不带 prefetch ----------------
print("=== Test 1: 不带 prefetch ===")
out = run_op()
check_out("[T1]", out)
print("=== Test 1 PASS ===")

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
staging, counter, slot_map = make_prefetch_tensors()
out2a = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=1e30, token_size_bytes_k=token_size_bytes_k)
check_out("[T2a]", out2a)
print("[T2a] used slots:", int((slot_map != -1).sum().item()), "(应为 0)")
print("=== Test 2a PASS ===")

# ---------------- Test 2b: 真实 prefetch ----------------
# 阈值调参: 随机数据下 score 标准差约 ~6e4, 200000 约 3σ 只触发少量 token。
# kernel 内 bitPositions 数组上限 256: 单 tile 超阈值 token 数须远小于 256,
# used=0 则降到 100000, 崩了则升到 1000000
print("=== Test 2b: 真实 prefetch ===")
staging, counter, slot_map = make_prefetch_tensors()
out2b = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=200000.0, token_size_bytes_k=token_size_bytes_k)
check_out("[T2b]", out2b)
used = int((slot_map[0] != -1).sum().item())
print("[T2b] used slots:", used)
if used > 0:
    tok = (slot_map[0] != -1).nonzero()[0].item()
    slot = slot_map[0, tok].item()
    k_ok = torch.equal(staging[slot, :token_size_bytes_k].cpu(), k_pool[tok].cpu())
    v_ok = torch.equal(staging[slot, token_size_bytes_k:].cpu(), v_pool[tok].cpu())
    print(f"[T2b] token {tok} -> slot {slot}: K bytes match={k_ok}, V bytes match={v_ok}")
print("=== Test 2b PASS ===")
