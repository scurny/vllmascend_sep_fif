from __future__ import annotations

import torch

from tools.liq_test_memory import SourcePools


def make_correctness_inputs(
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    requests, rows, heads, dim = 2, 4, 64, 128
    length, topk, block_size = 4096, 2048, 128
    query = torch.zeros((requests * rows, heads, dim), dtype=torch.int8)
    key = torch.zeros((requests, length, dim), dtype=torch.int8)
    expected = torch.empty((requests * rows, 1, topk), dtype=torch.int64)
    hits = torch.zeros((requests, length), dtype=torch.bool)
    for req in range(requests):
        selected = torch.arange(req * 1024, req * 1024 + topk)
        for row in range(rows):
            token = req * rows + row
            query[token, :, token] = 1
            key[req, selected, token] = 1
            hot = torch.arange(1536 + row * 4, 1540 + row * 4)
            key[req, hot, token] = 2
            hits[req, hot] = True
            # A higher-scoring future key must be excluded by the causal mask.
            if row < rows - 1:
                key[req, length - rows + row + 1, token] = 2
            expected[token, 0] = selected
    inputs = {
        "query": query,
        "key": key.reshape(-1, block_size, 1, dim),
        "weights": torch.ones((requests * rows, heads), dtype=torch.float16),
        "query_dequant_scale": torch.ones((requests * rows, heads), dtype=torch.float16),
        "key_dequant_scale": torch.ones((requests * length // block_size, block_size, 1), dtype=torch.float16),
        "actual_seq_lengths_query": torch.tensor([rows, requests * rows], dtype=torch.int32),
        "actual_seq_lengths_key": torch.full((requests,), length, dtype=torch.int32),
        "block_table": torch.arange(requests * length // block_size, dtype=torch.int32).reshape(requests, -1),
    }
    positions = torch.arange(requests * length).reshape(requests, length)
    return {name: tensor.to(device) for name, tensor in inputs.items()}, expected, hits, positions


def assert_topk_set(output: torch.Tensor, expected: torch.Tensor) -> None:
    assert output.dtype == torch.int32, output.dtype
    assert output.shape == expected.shape, (output.shape, expected.shape)
    # Ties inside the selected set allow arbitrary order, but its boundary is strict.
    actual = output.cpu().sort(dim=-1).values
    assert torch.equal(actual, expected), "LIQ TopK differs from the known selected set"


def assert_prefetch_payload(
    staging: torch.Tensor,
    slot_map: torch.Tensor,
    expected_k: torch.Tensor,
    expected_v: torch.Tensor,
    expected_hits: torch.Tensor,
    *,
    require_all: bool = True,
) -> None:
    mapping, payload = slot_map.cpu(), staging.cpu()
    assert torch.all((mapping >= -1) & (mapping < len(payload))), "invalid staging slot"
    hits = mapping >= 0
    assert not torch.any(hits & ~expected_hits), "unexpected request/token prefetched"
    if require_all:
        assert torch.equal(hits, expected_hits), "missing prefetch hits"
    slots = mapping[hits].long()
    assert slots.unique().numel() == slots.numel(), "live staging slots overlap"
    k_bytes = expected_k.shape[-1]
    assert torch.equal(payload[slots, :k_bytes], expected_k[hits]), "K payload differs"
    assert torch.equal(payload[slots, k_bytes:], expected_v[hits]), "V payload differs"


def run_correctness_cases(pools: SourcePools) -> None:
    inputs, expected, expected_hits, positions = make_correctness_inputs("npu")
    gva = pools.gva(positions).to("npu")
    entry_bytes = pools.k.shape[-1] + pools.v.shape[-1]
    threshold = 3 / 32  # Base score 1/16; hot score 1/8 after kernel's 2^-10 scaling.

    def launch(
        staging: torch.Tensor | None = None,
        slot_map: torch.Tensor | None = None,
        counter: torch.Tensor | None = None,
        ema_threshold: float = threshold,
    ) -> torch.Tensor:
        output = torch.ops._C_ascend.npu_lightning_indexer_quant(
            **inputs,
            query_quant_mode=0, key_quant_mode=0, layout_query="TND",
            layout_key="PA_BSND", sparse_count=2048, sparse_mode=3,
            token_gva_table=gva if staging is not None else None,
            prefetch_staging_buf=staging, prefetch_slot_counter=counter,
            token_to_prefetch_slot=slot_map, ema_threshold=ema_threshold,
            token_size_bytes_k=pools.k.shape[-1],
        )
        torch.npu.synchronize()
        assert_topk_set(output, expected)
        return output

    print("=== Test 7: 已知 TopK、因果掩码、多请求和源 KV 更新 ===")
    baseline = launch()
    slot_map = torch.full(expected_hits.shape, -1, dtype=torch.int32, device="npu")
    counter = torch.zeros(1, dtype=torch.int32, device="npu")
    staging = torch.full((4096, entry_bytes), 0xA5, dtype=torch.uint8, device="npu")
    launch(staging, slot_map, counter, ema_threshold=1e30)
    assert torch.all(slot_map == -1).item()
    assert torch.all(staging == 0xA5).item()
    for iteration in range(3):
        # The previous launch is synchronized before changing DDR/HBM source bytes.
        pools.k.add_(1)
        pools.v.add_(3)
        staging.fill_(0xA5)
        slot_map.fill_(-1)
        counter.zero_()
        output = launch(staging, slot_map, counter)
        assert torch.equal(output.cpu().sort().values, baseline.cpu().sort().values)
        assert_prefetch_payload(
            staging, slot_map, pools.k.cpu()[positions], pools.v.cpu()[positions], expected_hits,
        )
        print(f"[T7 iteration {iteration}] TopK and {expected_hits.sum().item()} payloads match")
    print("=== Test 7 PASS ===")

    print("=== Test 8: 极小 staging 容量的安全回退、TopK 和保护区 ===")
    for capacity in (1, 2, 16):
        guarded = torch.full((capacity + 2, entry_bytes), 0xA5, dtype=torch.uint8, device="npu")
        staging = guarded[1:-1]
        slot_map.fill_(-1)
        counter.zero_()
        launch(staging, slot_map, counter)
        assert_prefetch_payload(
            staging, slot_map, pools.k.cpu()[positions], pools.v.cpu()[positions], expected_hits,
            require_all=False,
        )
        assert torch.all(guarded[[0, -1]] == 0xA5).item(), "staging guard overwritten"
        used = (slot_map >= 0).sum().item()
        assert used < expected_hits.sum().item(), "capacity case did not exercise a miss"
        print(f"[T8 capacity {capacity}] {used} hits; known TopK and guards match")
    print("=== Test 8 PASS ===")
