# SPDX-License-Identifier: Apache-2.0
import torch

from tools.liq_test_diagnostics import T2bFailureDiagnostics, t2b_failure_lines


def make_diagnostics(staging: torch.Tensor, slot_map: torch.Tensor) -> T2bFailureDiagnostics:
    return T2bFailureDiagnostics(
        staging=staging,
        slot_map=slot_map,
        output=torch.tensor([[[0, 256, 257, 258, 259]]], dtype=torch.int32),
        query=torch.ones((1, 64, 2), dtype=torch.int8),
        key=torch.zeros((300, 2), dtype=torch.int8),
        weights=torch.ones((1, 64), dtype=torch.float16),
        query_dequant_scale=torch.ones((1, 64), dtype=torch.float16),
        key_dequant_scale=torch.ones((300,), dtype=torch.float16),
        threshold=1 / 32,
        hot_tokens=torch.arange(256, 260),
    )


def test_t2b_diagnostics_reports_no_payload_write() -> None:
    lines = t2b_failure_lines(
        make_diagnostics(torch.full((8, 3), 0xA5, dtype=torch.uint8), torch.full((1, 300), -1, dtype=torch.int32))
    )

    assert any("no payload write observed" in line for line in lines)
    assert any("topk_hot_positions" in line and "256:1" in line for line in lines)


def test_t2b_diagnostics_distinguishes_payload_without_slot_map() -> None:
    staging = torch.full((8, 3), 0xA5, dtype=torch.uint8)
    staging[0, 0] = 0
    lines = t2b_failure_lines(make_diagnostics(staging, torch.full((1, 300), -1, dtype=torch.int32)))

    assert any("payload changed but slot map has no live entries" in line for line in lines)


def test_t2b_diagnostics_keeps_fractional_scale_checksum() -> None:
    diagnostics = make_diagnostics(
        torch.full((8, 3), 0xA5, dtype=torch.uint8), torch.full((1, 300), -1, dtype=torch.int32)
    )
    diagnostics.query_dequant_scale.fill_(0.5)

    assert any("query_scale=32.0" in line for line in t2b_failure_lines(diagnostics))


def test_t2b_diagnostics_handles_scalar_topk_output() -> None:
    diagnostics = make_diagnostics(
        torch.zeros((8, 3), dtype=torch.uint8), torch.full((1, 300), -1, dtype=torch.int32)
    )
    diagnostics = T2bFailureDiagnostics(
        staging=diagnostics.staging,
        slot_map=diagnostics.slot_map,
        output=torch.tensor(256, dtype=torch.int32),
        query=diagnostics.query,
        key=diagnostics.key,
        weights=diagnostics.weights,
        query_dequant_scale=diagnostics.query_dequant_scale,
        key_dequant_scale=diagnostics.key_dequant_scale,
        threshold=diagnostics.threshold,
        hot_tokens=diagnostics.hot_tokens,
    )

    lines = t2b_failure_lines(diagnostics)

    assert any("topk_shape=()" in line and "256:0" in line for line in lines)
