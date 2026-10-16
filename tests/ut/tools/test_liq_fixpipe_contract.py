# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CUBE_HEADER = ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/"
    "lightning_indexer_quant_service_cube_arch35.h"
)


def _function(source: str, name: str) -> str:
    start = source.index(f"LIQMatmul<LIQT>::{name}")
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated {name}")


def test_prequant_fixpipe_barrier_precedes_l1_copy() -> None:
    fixp_s_to_l1 = _function(CUBE_HEADER.read_text(encoding="utf-8"), "FixpSToL1")
    prequant = fixp_s_to_l1.index("SetFixpipePreQuantFlag")
    barrier = fixp_s_to_l1.index("PipeBarrier<PIPE_FIX>", prequant)
    copy = fixp_s_to_l1.index("DataCopy(", prequant)
    assert prequant < barrier < copy
