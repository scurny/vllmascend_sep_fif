# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from tools.liq_test_correctness import assert_topk_set, make_correctness_inputs


def run_workspace_cases() -> None:
    inputs, expected, _, _ = make_correctness_inputs("npu")
    outputs = []
    sentinels = []
    print("=== Workspace: queued calls and allocator reuse ===")
    for _ in range(4):
        outputs.append(torch.ops._C_ascend.npu_lightning_indexer_quant(
            **inputs, query_quant_mode=0, key_quant_mode=0,
            layout_query="TND", layout_key="PA_BSND", sparse_count=2048, sparse_mode=3,
        ))
        # Keep allocations live while subsequent LIQ commands are queued.
        sentinel = torch.full((32 * 1024 * 1024,), 0xA5, dtype=torch.uint8, device="npu")
        sentinels.append(sentinel)
    torch.npu.synchronize()
    for output in outputs:
        assert_topk_set(output, expected)
    for sentinel in sentinels:
        assert torch.all(sentinel == 0xA5).item(), "LIQ workspace overwrote a live allocation"
    print("=== Workspace PASS ===")
