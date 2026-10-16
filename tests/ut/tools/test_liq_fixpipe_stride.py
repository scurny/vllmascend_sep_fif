# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CUBE_HEADER = ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/"
    "lightning_indexer_quant_service_cube_arch35.h"
)


def _fixpipe_result_to_gm() -> str:
    source = CUBE_HEADER.read_text(encoding="utf-8")
    start = source.index("LIQMatmul<LIQT>::FixpResToGm")
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError("unterminated FixpResToGm")


def _stride_expressions(fixpipe: str) -> tuple[str, str]:
    match = re.search(
        r"#if defined\(__CCE_AICORE__\) && __CCE_AICORE__ == 310"
        r"(?P<arch310>.*?)#else(?P<arch220>.*?)#endif",
        fixpipe,
        re.DOTALL,
    )
    assert match is not None

    def assignment(branch: str) -> str:
        stride = re.search(r"srcNdStride\s*=\s*(?P<value>[^;]+);", branch)
        assert stride is not None
        return stride.group("value")

    return assignment(match.group("arch310")), assignment(match.group("arch220"))


def _compile_contract(tmp_path: Path, arch310: str, arch220: str, architecture: int) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to execute the Fixpipe stride contract")
    source = tmp_path / f"fixpipe_{architecture}.cc"
    binary = source.with_suffix("")
    source.write_text(
        "#include <cstdint>\n"
        "#include <initializer_list>\n"
        "constexpr uint64_t BLOCK_CUBE = 16, S2_BASIC_BLOCK_L0 = 128;\n"
        "constexpr uint64_t CeilDiv(uint64_t x, uint64_t y) { return (x + y - 1) / y; }\n"
        "struct ConstInfo { uint64_t gSize; };\n"
        "int main() {\n"
        "  const ConstInfo constInfo_ {64};\n"
        "#if __CCE_AICORE__ == 310\n"
        f"  const uint64_t srcNdStride = {arch310};\n"
        "  constexpr uint64_t strideUnitElements = BLOCK_CUBE;\n"
        "#else\n"
        f"  const uint64_t srcNdStride = {arch220};\n"
        "  constexpr uint64_t strideUnitElements = 256;\n"
        "#endif\n"
        "  for (uint64_t rows : {1ULL, 2ULL, 4ULL}) {\n"
        "    const uint64_t computeWsElements = rows * constInfo_.gSize * S2_BASIC_BLOCK_L0;\n"
        "    if (rows * srcNdStride * strideUnitElements != computeWsElements) return 1;\n"
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [compiler, "-std=c++17", f"-D__CCE_AICORE__={architecture}", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("architecture", [220, 310])
def test_fixpipe_nz_stride_matches_compute_ws_rows(tmp_path: Path, architecture: int) -> None:
    fixpipe = _fixpipe_result_to_gm()
    arch310, arch220 = _stride_expressions(fixpipe)
    call = re.search(r"SetFixpipeNz2ndFlag\((?P<args>.*?)\);", fixpipe, re.DOTALL)
    assert call is not None
    assert re.search(r",\s*srcNdStride\s*,", call.group("args")) is not None
    binary = _compile_contract(tmp_path, arch310, arch220, architecture)
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
