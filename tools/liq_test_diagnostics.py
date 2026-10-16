# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class T2bFailureDiagnostics:
    staging: torch.Tensor
    slot_map: torch.Tensor
    output: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    weights: torch.Tensor
    query_dequant_scale: torch.Tensor
    key_dequant_scale: torch.Tensor
    threshold: float
    hot_tokens: torch.Tensor


def _checksum(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu().to(torch.float32).sum().item())


def _topk_hot_positions(output: torch.Tensor, hot_tokens: torch.Tensor) -> str:
    flat_output = output.detach().cpu().reshape(-1)
    positions: list[str] = []
    for hot_token in hot_tokens.detach().cpu().reshape(-1):
        token = int(hot_token.item())
        matches = (flat_output == token).nonzero().reshape(-1)
        locations = ",".join(str(int(position.item())) for position in matches[:4])
        positions.append(f"{token}:{locations or '-'}")
    return " ".join(positions)


def t2b_failure_lines(diagnostics: T2bFailureDiagnostics) -> list[str]:
    """Inspect T2b outputs after synchronization; staging starts filled with 0xA5."""
    staging = diagnostics.staging.detach().cpu()
    slot_map = diagnostics.slot_map.detach().cpu().reshape(-1)
    hot_tokens = diagnostics.hot_tokens.detach().cpu().reshape(-1)
    changed_rows = int((staging != 0xA5).any(dim=-1).sum().item()) if staging.ndim >= 2 else 0
    live_slots = int((slot_map >= 0).sum().item())
    hot_map: list[str] = []
    for hot_token in hot_tokens:
        token = int(hot_token.item())
        mapped_slot = int(slot_map[token].item()) if token < slot_map.numel() else -2
        hot_map.append(f"{token}:{mapped_slot}")

    key_rows = diagnostics.key.detach().cpu().reshape(-1, diagnostics.key.shape[-1])
    hot_key = key_rows[hot_tokens.long()] if hot_tokens.numel() else key_rows[:0]
    zero_control = key_rows[:1]
    lines = [
        "[T2b diag] observation only; this does not identify the root cause",
        f"[T2b diag] threshold={diagnostics.threshold:.6f} staging_changed_rows={changed_rows} "
        f"live_slots={live_slots} hot_map={' '.join(hot_map)}",
        f"[T2b diag] topk_shape={tuple(diagnostics.output.shape)} "
        f"topk_hot_positions={_topk_hot_positions(diagnostics.output, hot_tokens)}",
        f"[T2b diag] checksums query={_checksum(diagnostics.query)} hot_key={_checksum(hot_key)} "
        f"zero_control_key={_checksum(zero_control)} weights={_checksum(diagnostics.weights)} "
        f"query_scale={_checksum(diagnostics.query_dequant_scale)} "
        f"key_scale={_checksum(diagnostics.key_dequant_scale)}",
    ]
    if changed_rows == 0:
        lines.append("[T2b diag] observation: staging has no changed rows; no payload write observed")
    elif live_slots == 0:
        lines.append("[T2b diag] observation: staging payload changed but slot map has no live entries")
    else:
        lines.append("[T2b diag] observation: payload and slot-map writes were observed; inspect token/slot matching")
    return lines
