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


def make_expected_score_lookup(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    query = inputs["query"].cpu().float()
    weights = inputs["weights"].cpu().float()
    query_scale = inputs["query_dequant_scale"].cpu().float()
    query_ends = inputs["actual_seq_lengths_query"].cpu().tolist()
    key_ends = inputs["actual_seq_lengths_key"].cpu().tolist()
    requests = len(key_ends)
    keys = inputs["key"].cpu().reshape(requests, -1, query.shape[-1]).float()
    key_scales = inputs["key_dequant_scale"].cpu().reshape(requests, -1).float()

    request_scores = []
    query_start = 0
    for request, (query_end, key_end) in enumerate(zip(query_ends, key_ends)):
        request_query = query[query_start:query_end]
        dot_products = torch.einsum("thd,sd->ths", request_query, keys[request, :key_end])
        scaled_scores = dot_products.relu()
        scaled_scores.mul_(query_scale[query_start:query_end, :, None])
        scaled_scores.mul_(key_scales[request, None, None, :key_end])
        scaled_scores.mul_(weights[query_start:query_end, :, None])
        request_scores.append(scaled_scores.sum(dim=1).div_(1024))
        query_start = query_end
    return torch.cat(request_scores)


def assert_topk_scores(
    indices: torch.Tensor,
    values: torch.Tensor,
    score_lookup: torch.Tensor,
) -> None:
    assert indices.dtype == torch.int32, indices.dtype
    assert values.dtype == torch.float32, values.dtype
    assert values.shape == indices.shape, (values.shape, indices.shape)
    assert score_lookup.ndim == 2, score_lookup.shape
    assert score_lookup.shape[0] == indices.shape[0], (score_lookup.shape, indices.shape)

    actual_indices = indices.cpu()
    actual_values = values.cpu()
    assert torch.all(actual_indices >= -1), "TopK index is below the padding sentinel"
    valid = actual_indices >= 0
    assert torch.all(actual_indices[valid] < score_lookup.shape[1]), "TopK index exceeds KV length"
    assert torch.all(torch.isfinite(actual_values[valid])), "valid TopK score is not finite"
    assert torch.all(actual_values[~valid] == -torch.inf), "padding score must be -inf"

    expected_values = score_lookup.gather(
        1, actual_indices.clamp_min(0).long().reshape(indices.shape[0], -1),
    ).reshape_as(actual_values)
    assert torch.allclose(actual_values[valid], expected_values[valid]), "TopK score differs from its index"


def assert_sequence_padded_scores(
    indices: torch.Tensor,
    values: torch.Tensor,
    score_lookup: torch.Tensor,
    sequence_lengths: list[int],
) -> None:
    assert_topk_scores(indices, values, score_lookup)
    assert not torch.any(torch.isnan(values.cpu())), "score output contains uninitialized NaN"
    assert len(sequence_lengths) == indices.shape[0], (sequence_lengths, indices.shape)
    for row, sequence_length in enumerate(sequence_lengths):
        row_indices = indices[row].cpu()
        valid = row_indices >= 0
        assert valid.sum().item() == min(sequence_length, indices.shape[-1]), "wrong valid TopK count"
        assert torch.all(row_indices[valid] < sequence_length), "TopK index exceeds sequence length"


def run_score_padding_cases(
    inputs: dict[str, torch.Tensor],
    expected_score_lookup: torch.Tensor,
) -> None:
    query_rows = torch.tensor([0, 4], device="npu")
    case_inputs = dict(inputs)
    case_inputs.update({
        "query": inputs["query"][query_rows],
        "weights": inputs["weights"][query_rows],
        "query_dequant_scale": inputs["query_dequant_scale"][query_rows],
        "actual_seq_lengths_query": torch.tensor([1, 2], dtype=torch.int32, device="npu"),
    })
    case_score_lookup = expected_score_lookup[[0, 4]]
    for actual_seq_lengths_key in ([65, 65], [0, 0], [0, 65]):
        case_inputs["actual_seq_lengths_key"] = torch.tensor(
            actual_seq_lengths_key, dtype=torch.int32, device="npu",
        )
        sparse_values_out = torch.full(
            (2, 1, 2048), float("nan"), dtype=torch.float32, device="npu",
        )
        indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
            **case_inputs,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
            sparse_values_out=sparse_values_out,
        )
        torch.npu.synchronize()
        assert_sequence_padded_scores(
            indices, sparse_values_out, case_score_lookup, actual_seq_lengths_key,
        )
        print(f"[T7 padding {actual_seq_lengths_key}] score padding matches indices")


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
    expected_score_lookup = make_expected_score_lookup(inputs)
    gva = pools.gva(positions).to("npu")
    entry_bytes = pools.k.shape[-1] + pools.v.shape[-1]
    threshold = 3 / 32  # Base score 1/16; hot score 1/8 after kernel's 2^-10 scaling.

    def launch(
        staging: torch.Tensor | None = None,
        slot_map: torch.Tensor | None = None,
        counter: torch.Tensor | None = None,
        ema_threshold: float = threshold,
    ) -> torch.Tensor:
        sparse_values_out = torch.full(
            expected.shape, float("nan"), dtype=torch.float32, device="npu",
        )
        output = torch.ops._C_ascend.npu_lightning_indexer_quant(
            **inputs,
            query_quant_mode=0, key_quant_mode=0, layout_query="TND",
            layout_key="PA_BSND", sparse_count=2048, sparse_mode=3,
            token_gva_table=gva if staging is not None else None,
            prefetch_staging_buf=staging, prefetch_slot_counter=counter,
            token_to_prefetch_slot=slot_map, ema_threshold=ema_threshold,
            token_size_bytes_k=pools.k.shape[-1],
            sparse_values_out=sparse_values_out,
        )
        torch.npu.synchronize()
        assert_topk_set(output, expected)
        assert_topk_scores(output, sparse_values_out, expected_score_lookup)
        return output

    print("=== Test 7a: TopK score padding for short and invalid sequences ===")
    run_score_padding_cases(inputs, expected_score_lookup)
    print("=== Test 7a PASS ===")

    print("=== Test 7: 已知 TopK、因果掩码、多请求和源 KV 更新 ===")
    baseline = launch()
    slot_map = torch.full(expected_hits.shape, -1, dtype=torch.int32, device="npu")
    counter = torch.zeros(1, dtype=torch.int32, device="npu")
    staging = torch.full((4096, entry_bytes), 0xA5, dtype=torch.uint8, device="npu")
    for ema_threshold in (-1.0, -2.0, -3.0, 0.0):
        launch(ema_threshold=ema_threshold)
        launch(staging, slot_map, counter, ema_threshold=ema_threshold)
        assert torch.all(slot_map == -1).item()
        assert torch.all(staging == 0xA5).item()
        assert torch.all(counter == 0).item()
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
