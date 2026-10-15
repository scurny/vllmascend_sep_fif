import torch
import torch_npu
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
    assert out.shape == (B * S1, 1, topk), (tag, out.shape)
    assert out.dtype == torch.int32, (tag, out.dtype)
    assert torch.all((out >= -1) & (out < asl_k)).item(), tag
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
    _out = run_op(ema_threshold=-_s)
    torch.npu.synchronize()
    print(f"=== STAGE {_s:g} OK (无 aicore 错误) ===")
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
query.fill_(1)
key.zero_()
key.view(-1, D)[256:320].fill_(1)
weights.fill_(1)
query_dequant_scale.fill_(1)
key_dequant_scale.fill_(1)
print("=== Test 2a: prefetch 开启, 阈值巨大 (零触发) ===")
staging, counter, slot_map = make_prefetch_tensors()
out2a = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=1e30, token_size_bytes_k=token_size_bytes_k)
check_out("[T2a]", out2a)
print("[T2a] used slots:", int((slot_map != -1).sum().item()), "(应为 0)")
assert torch.all(slot_map == -1).item()
print("=== Test 2a PASS ===")

# ---------------- Test 2b: 真实 prefetch ----------------
print("=== Test 2b: 真实 prefetch ===")
staging, counter, slot_map = make_prefetch_tensors()
out2b = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=1.0, token_size_bytes_k=token_size_bytes_k)
check_out("[T2b]", out2b)
used = int((slot_map[0] != -1).sum().item())
print("[T2b] used slots:", used)
assert used == 64, used
hit_tokens = (slot_map[0].cpu() >= 0).nonzero().flatten()
assert torch.equal(hit_tokens, torch.arange(256, 320))
hit_slots = slot_map[0].cpu()[hit_tokens].long()
assert torch.unique(hit_slots).numel() == used
assert torch.equal(staging.cpu()[hit_slots, :token_size_bytes_k], k_pool.cpu()[hit_tokens])
assert torch.equal(staging.cpu()[hit_slots, token_size_bytes_k:], v_pool.cpu()[hit_tokens])
print("=== Test 2b PASS ===")

print("=== Test 3: 拒绝非法 K 字节数 ===")
for invalid_k_bytes in (-1, 1, k_entry_bytes, k_entry_bytes + 1, 512, (1 << 32) + 256):
    rejected = False
    try:
        run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
               prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
               ema_threshold=1.0, token_size_bytes_k=invalid_k_bytes)
    except RuntimeError:
        rejected = True
    assert rejected, f"invalid token_size_bytes_k accepted: {invalid_k_bytes}"
torch.npu.synchronize()
staging, counter, slot_map = make_prefetch_tensors()
out3 = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
              prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map, ema_threshold=1.0)
check_out("[T3 default K]", out3)
assert torch.equal((slot_map[0].cpu() >= 0).nonzero().flatten(), hit_tokens)
default_slots = slot_map[0].cpu()[hit_tokens].long()
assert torch.equal(staging.cpu()[default_slots, :token_size_bytes_k], k_pool.cpu()[hit_tokens])
assert torch.equal(staging.cpu()[default_slots, token_size_bytes_k:], v_pool.cpu()[hit_tokens])
print("=== Test 3 PASS ===")
