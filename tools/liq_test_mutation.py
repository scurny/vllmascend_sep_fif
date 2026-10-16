# SPDX-License-Identifier: Apache-2.0
import torch

from tools.liq_test_correctness import assert_prefetch_payload, assert_topk_set, make_correctness_inputs
from tools.liq_test_memory import SourcePools


def run_mutation_cases(pools: SourcePools) -> None:
    inputs, expected, hits, positions = make_correctness_inputs("npu")
    entry_bytes = pools.k.shape[-1] + pools.v.shape[-1]
    staging = torch.full((4096, entry_bytes), 0xA5, dtype=torch.uint8, device="npu")
    slot_map = torch.full(hits.shape, -1, dtype=torch.int32, device="npu")
    prefetch = {
        "token_gva_table": pools.gva(positions).to("npu"),
        "prefetch_staging_buf": staging,
        "prefetch_slot_counter": torch.zeros(1, dtype=torch.int32, device="npu"),
        "token_to_prefetch_slot": slot_map,
    }

    def launch(buffers: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.ops._C_ascend.npu_lightning_indexer_quant(
            **inputs, **buffers, query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND", sparse_count=2048, sparse_mode=3,
            ema_threshold=3 / 32, token_size_bytes_k=pools.k.shape[-1],
        )

    print("=== Writable buffers: layout and device validation ===")
    strided_staging = torch.empty((4096, entry_bytes * 2), dtype=torch.uint8, device="npu")[:, ::2]
    strided_map = torch.empty((hits.shape[0], hits.shape[1] * 2), dtype=torch.int32, device="npu")[:, ::2]
    invalid = [
        ("prefetch_staging_buf", strided_staging, "must be contiguous"),
        ("token_to_prefetch_slot", strided_map, "must be contiguous"),
    ]
    invalid.extend((name, tensor.cpu(), "must be on the same device") for name, tensor in prefetch.items())
    for name, tensor, message in invalid:
        try:
            launch(prefetch | {name: tensor})
        except RuntimeError as error:
            assert name in str(error) and message in str(error), str(error)
        else:
            raise AssertionError(f"invalid {name} accepted")
    torch.npu.synchronize()
    assert torch.all(staging == 0xA5).item()
    assert torch.all(slot_map == -1).item()
    output = launch(prefetch)
    torch.npu.synchronize()
    assert_topk_set(output, expected)
    assert_prefetch_payload(staging, slot_map, pools.k.cpu()[positions], pools.v.cpu()[positions], hits)
    print("=== Writable buffers PASS ===")
