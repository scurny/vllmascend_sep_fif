# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant.cpp"
VECTOR = ROOT / "csrc/attention/lightning_indexer_quant/op_kernel/arch35/lightning_indexer_quant_service_vector_arch35.h"


def test_prefetch_device_diagnostics_are_opt_in_and_bounded() -> None:
    kernel = KERNEL.read_text()
    vector = VECTOR.read_text()

    assert "#if defined(LIQ_PREFETCH_DIAGNOSTICS) && LIQ_PREFETCH_DIAGNOSTICS" in kernel
    assert '#include "utils/debug/asc_printf.h"' in kernel
    assert '#include "utils/debug/asc_printf.h"' in vector
    assert "static_cast<int32_t>((gvaTable) != nullptr)" in kernel
    assert "static_cast<int32_t>(cuS2Len > 256)" in vector
    assert "entry v1 built=%s %s" in kernel
    assert "gva=%d staging=%d counter=%d " in kernel
    assert '"slot=%d\\n"' in kernel
    assert "GetBlockIdx() == 0" in kernel
    assert "prefetch skipped reason=capacity" in vector
    assert "prefetch skipped reason=too_many_candidates" in vector
    assert "prefetch skipped reason=row_capacity" in vector
    assert "direct=%u mask=%u" in vector
    assert "mask_byte32_valid=%d" in vector
    assert "scores255_260=" in vector
    assert "nan=%u inf=%u" in vector
    assert "map_write=%s" in vector
    assert "(void)isFirstLogicalRow;" in vector
    assert "const bool isFirstLogicalRow = false;" in vector
    assert "blockId_ == 0 && info.bIdx == 0 && info.gS1Idx == 0 && info.s2Idx == 0 && innerS1Idx == 0" in vector


def _preprocess(source: str, enabled: int) -> str:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ preprocessor is required")
    source_without_includes = "\n".join(
        line for line in source.splitlines() if not line.startswith("#include")
    )
    completed = subprocess.run(
        [
            compiler,
            "-E",
            "-P",
            "-x",
            "c++",
            "-D__CCE_AICORE__=310",
            f"-DLIQ_PREFETCH_DIAGNOSTICS={enabled}",
            "-",
        ],
        input=source_without_includes,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout


def test_diagnostic_prints_are_absent_when_the_build_flag_is_zero() -> None:
    source = KERNEL.read_text() + "\n" + VECTOR.read_text()

    disabled = _preprocess(source, 0)
    enabled = _preprocess(source, 1)

    assert "LIQ_DIAG" not in disabled
    assert "LIQ_DIAG" in enabled
