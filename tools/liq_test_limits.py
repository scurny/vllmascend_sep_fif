# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

from tools.liq_test_correctness import assert_topk_set, make_correctness_inputs


def expect_host_rejection(label: str, launch: Callable[[], torch.Tensor]) -> None:
    try:
        launch()
    except RuntimeError:
        print(f"{label}: synchronous rejection confirmed")
    else:
        raise AssertionError(f"{label} was accepted by the LIQ host tiling validation")


def run_limits_cases() -> None:
    inputs, expected, _, _ = make_correctness_inputs("npu")

    def launch_normal() -> torch.Tensor:
        return torch.ops._C_ascend.npu_lightning_indexer_quant(
            **inputs,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        )

    output = launch_normal()
    torch.npu.synchronize()
    assert_topk_set(output, expected)
    print("normal LIQ path: confirmed")

    head_inputs = inputs.copy()
    head_inputs["query"] = head_inputs["query"][:, :32]
    head_inputs["weights"] = head_inputs["weights"][:, :32]
    head_inputs["query_dequant_scale"] = head_inputs["query_dequant_scale"][:, :32]
    expect_host_rejection(
        "query heads=32",
        lambda: torch.ops._C_ascend.npu_lightning_indexer_quant(
            **head_inputs,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
        ),
    )

    gva = torch.zeros((2, 4096, 2), dtype=torch.int64, device="npu")
    staging = torch.empty((1, 131_072), dtype=torch.uint8, device="npu")
    counter = torch.zeros(1, dtype=torch.int32, device="npu")
    slot_map = torch.full((2, 4096), -1, dtype=torch.int32, device="npu")
    expect_host_rejection(
        "prefetch entry_bytes=131072",
        lambda: torch.ops._C_ascend.npu_lightning_indexer_quant(
            **inputs,
            query_quant_mode=0,
            key_quant_mode=0,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=2048,
            sparse_mode=3,
            token_gva_table=gva,
            prefetch_staging_buf=staging,
            prefetch_slot_counter=counter,
            token_to_prefetch_slot=slot_map,
            ema_threshold=1.0,
            token_size_bytes_k=65_536,
        ),
    )
    output = launch_normal()
    torch.npu.synchronize()
    assert_topk_set(output, expected)
    print("=== Host limits PASS ===")
