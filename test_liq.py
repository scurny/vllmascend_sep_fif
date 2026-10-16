import argparse
import os
import sys

import torch
import torch_npu

from tools.liq_test_correctness import run_correctness_cases
from tools.liq_test_memory import make_source_pools
from vllm_ascend import vllm_ascend_C  # noqa: F401  注册 torch.ops

parser = argparse.ArgumentParser(description="LIQ TopK and KV prefetch correctness tests")
parser.add_argument("--ddr", action="store_true", help="Use registered memfabric DDR K/V source pools")
args = parser.parse_args()
torch.manual_seed(0)
PREFETCH_HIT_THRESHOLD = 1 / 32  # Single-dimension score: 64 heads * 2^-10 = 1/16.

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

pools = make_source_pools(kSeqSize, token_size_bytes_k, k_entry_bytes - token_size_bytes_k, ddr=args.ddr)
k_pool, v_pool = pools.k, pools.v
print(f"[LIQ] K/V source: {'registered DDR (DVA)' if args.ddr else 'HBM'}")
token_gva_table = pools.gva(torch.arange(kSeqSize).view(B, -1)).to('npu')


def make_prefetch_tensors():
    return (torch.zeros((prefetch_candidate_max, k_entry_bytes), dtype=torch.uint8, device='npu'),
            torch.zeros(1, dtype=torch.int32, device='npu'),
            torch.full((B, kSeqSize), -1, dtype=torch.int32, device='npu'))


# ---------------- Test 2a: prefetch 开启但零触发 ----------------
prefetch_hits_per_row = 4
query.fill_(1)
key.zero_()
key.view(-1, D)[256:256 + prefetch_hits_per_row].fill_(1)
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
               ema_threshold=PREFETCH_HIT_THRESHOLD, token_size_bytes_k=token_size_bytes_k)
check_out("[T2b]", out2b)
used = int((slot_map[0] != -1).sum().item())
print("[T2b] used slots:", used)
assert used == prefetch_hits_per_row, used
hit_tokens = (slot_map[0].cpu() >= 0).nonzero().flatten()
assert torch.equal(hit_tokens, torch.arange(256, 256 + prefetch_hits_per_row))
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
               ema_threshold=PREFETCH_HIT_THRESHOLD, token_size_bytes_k=invalid_k_bytes)
    except RuntimeError:
        rejected = True
    assert rejected, f"invalid token_size_bytes_k accepted: {invalid_k_bytes}"
torch.npu.synchronize()
staging, counter, slot_map = make_prefetch_tensors()
out3 = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
              prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map, ema_threshold=PREFETCH_HIT_THRESHOLD)
check_out("[T3 default K]", out3)
assert torch.equal((slot_map[0].cpu() >= 0).nonzero().flatten(), hit_tokens)
default_slots = slot_map[0].cpu()[hit_tokens].long()
assert torch.equal(staging.cpu()[default_slots, :token_size_bytes_k], k_pool.cpu()[hit_tokens])
assert torch.equal(staging.cpu()[default_slots, token_size_bytes_k:], v_pool.cpu()[hit_tokens])
print("=== Test 3 PASS ===")

print("=== Test 4: 多行预取槽位不覆盖 ===")
S1 = 4
query = torch.zeros((S1, N1, D), dtype=torch.int8, device='npu')
weights = torch.ones((S1, N1), dtype=torch.float16, device='npu')
query_dequant_scale = torch.ones_like(weights)
actual_seq_lengths_query = torch.tensor([S1], dtype=torch.int32, device='npu')
key.zero_()
for row in range(S1):
    query[row, :, row] = 1
    start = 256 + row * prefetch_hits_per_row
    key.view(-1, D)[start:start + prefetch_hits_per_row, row] = 1
expected_tokens = torch.arange(256, 256 + S1 * prefetch_hits_per_row)
staging, counter, slot_map = make_prefetch_tensors()
for iteration in range(8):
    staging.fill_(0xA5)
    counter.zero_()
    slot_map.fill_(-1)
    out4 = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
                  prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
                  ema_threshold=PREFETCH_HIT_THRESHOLD, token_size_bytes_k=token_size_bytes_k)
    check_out(f"[T4 iteration {iteration}]", out4)
    hit_tokens = (slot_map[0].cpu() >= 0).nonzero().flatten()
    assert torch.equal(hit_tokens, expected_tokens)
    hit_slots = slot_map[0].cpu()[hit_tokens].long()
    assert torch.unique(hit_slots).numel() == expected_tokens.numel()
    assert torch.equal(staging.cpu()[hit_slots, :token_size_bytes_k], k_pool.cpu()[hit_tokens])
    assert torch.equal(staging.cpu()[hit_slots, token_size_bytes_k:], v_pool.cpu()[hit_tokens])
print("=== Test 4 PASS ===")

print("=== Test 5: 多核重复 token 与缓存行边界 ===")
query.zero_()
query[:, :, 0] = 1
key.zero_()
expected_tokens = torch.tensor([255, 256, 271, 272], dtype=torch.int64)
key.view(-1, D)[expected_tokens.to('npu'), 0] = 1
for iteration in range(8):
    staging.fill_(0xA5)
    counter.zero_()
    slot_map.fill_(-1)
    out5 = run_op(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
                  prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
                  ema_threshold=PREFETCH_HIT_THRESHOLD, token_size_bytes_k=token_size_bytes_k)
    check_out(f"[T5 iteration {iteration}]", out5)
    hit_tokens = (slot_map[0].cpu() >= 0).nonzero().flatten()
    assert torch.equal(hit_tokens, expected_tokens)
    hit_slots = slot_map[0].cpu()[hit_tokens].long()
    assert torch.unique(hit_slots).numel() == expected_tokens.numel()
    assert torch.equal(staging.cpu()[hit_slots, :token_size_bytes_k], k_pool.cpu()[hit_tokens])
    assert torch.equal(staging.cpu()[hit_slots, token_size_bytes_k:], v_pool.cpu()[hit_tokens])
print("=== Test 5 PASS ===")

print("=== Test 6: 拒绝错误的预取表形状 ===")
invalid_tables = [
    ("GVA rank", "token_gva_table", token_gva_table.reshape(-1, 2)),
    ("GVA rows", "token_gva_table", token_gva_table[:0]),
    ("GVA short width", "token_gva_table", token_gva_table[:, :-1].contiguous()),
    ("GVA long width", "token_gva_table",
     torch.zeros((B, kSeqSize + 1, 2), dtype=torch.int64, device='npu')),
    ("GVA address pair", "token_gva_table", token_gva_table[:, :, :1].contiguous()),
    ("map rank", "token_to_prefetch_slot", slot_map.reshape(-1)),
    ("map rows", "token_to_prefetch_slot", slot_map[:0]),
    ("map short width", "token_to_prefetch_slot", slot_map[:, :-1].contiguous()),
    ("map long width", "token_to_prefetch_slot",
     torch.full((B, kSeqSize + 1), -1, dtype=torch.int32, device='npu')),
    ("counter rank", "prefetch_slot_counter", counter.reshape(1, 1)),
    ("counter empty", "prefetch_slot_counter", counter[:0]),
]
for case_name, argument, invalid_tensor in invalid_tables:
    inputs = dict(token_gva_table=token_gva_table, prefetch_staging_buf=staging,
                  prefetch_slot_counter=counter, token_to_prefetch_slot=slot_map,
                  ema_threshold=PREFETCH_HIT_THRESHOLD, token_size_bytes_k=token_size_bytes_k)
    inputs[argument] = invalid_tensor
    rejected = False
    try:
        run_op(**inputs)
    except RuntimeError:
        rejected = True
    assert rejected, f"invalid prefetch shape accepted: {case_name}"
torch.npu.synchronize()

extra_rows_gva = token_gva_table.repeat(2, 1, 1)
extra_rows_map = torch.full((2 * B, kSeqSize), -1, dtype=torch.int32, device='npu')
out6 = run_op(token_gva_table=extra_rows_gva, prefetch_staging_buf=staging,
              prefetch_slot_counter=counter, token_to_prefetch_slot=extra_rows_map,
              ema_threshold=PREFETCH_HIT_THRESHOLD, token_size_bytes_k=token_size_bytes_k)
check_out("[T6 extra rows]", out6)
hit_tokens = (extra_rows_map[0].cpu() >= 0).nonzero().flatten()
assert torch.equal(hit_tokens, expected_tokens)
assert torch.all(extra_rows_map[B:] == -1).item()
hit_slots = extra_rows_map[0].cpu()[hit_tokens].long()
assert torch.equal(staging.cpu()[hit_slots, :token_size_bytes_k], k_pool.cpu()[hit_tokens])
assert torch.equal(staging.cpu()[hit_slots, token_size_bytes_k:], v_pool.cpu()[hit_tokens])
print("=== Test 6 PASS ===")

run_correctness_cases(pools)
torch.npu.synchronize()
