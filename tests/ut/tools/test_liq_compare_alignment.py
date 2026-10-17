# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
VECTOR = ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/"
    "lightning_indexer_quant_service_vector_arch35.h"
)
COUNTS = (1, 32, 33, 63, 64, 65, 2048)


def _compare_scalar_body(source: str) -> str:
    start = source.index("LIQVector<LIQT>::PrefetchScoresInVec1")
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError("unterminated PrefetchScoresInVec1")


def _extract_contract() -> tuple[str, str, str, str, str, str]:
    source = VECTOR.read_text(encoding="utf-8")
    body = _compare_scalar_body(source)
    candidate_section = body[body.index("// 收集超过阈值的 position") :]
    bytes_match = re.search(
        r"constexpr uint32_t COMPARE_SCALAR_ALIGNMENT_BYTES\s*=\s*(?P<value>[^;]+);",
        source,
    )
    floats_match = re.search(
        r"constexpr uint32_t COMPARE_SCALAR_ALIGNMENT_FLOATS\s*=\s*(?P<value>[^;]+);",
        source,
    )
    align_match = re.search(
        r"(?:const )?uint32_t cmpLenAlign\s*=\s*(?P<value>[^;]+);",
        body,
    )
    candidate_bound_match = re.search(
        r"for \(uint32_t j = 0; j < (?P<value>[^;]+); j\+\+\) \{",
        candidate_section,
    )
    buffer_match = re.search(
        r"pipe->InitBuffer\(cmpMaskBuf_,\s*(?P<value>[^;]+)\);",
        source,
    )
    input_buffer_match = re.search(
        r"pipe->InitBuffer\(inQueue_,\s*2,\s*(?P<value>[^;]+)\);",
        source,
    )
    assert bytes_match is not None
    assert floats_match is not None
    assert align_match is not None
    assert candidate_bound_match is not None
    assert "Duplicate(scoresUb[cuS2Len]" not in body
    assert buffer_match is not None
    assert input_buffer_match is not None
    return (
        bytes_match.group("value"),
        floats_match.group("value"),
        align_match.group("value"),
        candidate_bound_match.group("value"),
        buffer_match.group("value"),
        input_buffer_match.group("value"),
    )


def _compile_contract(tmp_path: Path) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to execute the CompareScalar alignment contract")

    (
        alignment_bytes,
        alignment_floats,
        align_expression,
        candidate_bound,
        buffer_expression,
        input_buffer_expression,
    ) = _extract_contract()
    source = tmp_path / "compare_alignment.cc"
    binary = source.with_suffix("")
    source.write_text(
        "#include <cstdint>\n"
        "#include <cstdlib>\n"
        "#include <iostream>\n"
        "#include <vector>\n"
        "namespace LIQCommon {\n"
        "constexpr uint32_t Align(uint32_t value, uint32_t alignment) {\n"
        "  return (value + alignment - 1U) / alignment * alignment;\n"
        "}\n"
        "constexpr uint32_t CeilDiv(uint32_t value, uint32_t divisor) {\n"
        "  return (value + divisor - 1U) / divisor;\n"
        "}\n"
        "}\n"
        f"constexpr uint32_t COMPARE_SCALAR_ALIGNMENT_BYTES = {alignment_bytes};\n"
        f"constexpr uint32_t COMPARE_SCALAR_ALIGNMENT_FLOATS = {alignment_floats};\n"
        "constexpr uint32_t S2_BASE_SIZE = 2048;\n"
        "int main(int argc, char **argv) {\n"
        "  const uint32_t cuS2Len = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));\n"
        f"  const uint32_t cmpLenAlign = {align_expression};\n"
        "  std::vector<uint8_t> cmp_mask((cmpLenAlign + 7U) / 8U, 0);\n"
        "  for (uint32_t index = cuS2Len; index < cmpLenAlign; ++index) {\n"
        "    cmp_mask[index / 8U] |= static_cast<uint8_t>(1U << (index % 8U));\n"
        "  }\n"
        "  uint32_t valid_candidates = 0;\n"
        f"  for (uint32_t j = 0; j < {candidate_bound}; ++j) {{\n"
        "    valid_candidates += (cmp_mask[j / 8U] >> (j % 8U)) & 1U;\n"
        "  }\n"
        f"  const uint32_t mask_bytes = {buffer_expression.replace('s2BaseSize_', 'S2_BASE_SIZE')};\n"
        f"  const uint32_t input_bytes = {input_buffer_expression.replace('s2BaseSize_', 'S2_BASE_SIZE')};\n"
        "  std::cout << cmpLenAlign << ' ' << mask_bytes << ' ' << valid_candidates << ' ' << input_bytes;\n"
        "}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("count", COUNTS)
def test_compare_scalar_uses_256_byte_alignment_and_ignores_poisoned_tail_mask_bits(tmp_path: Path, count: int) -> None:
    binary = _compile_contract(tmp_path)
    completed = subprocess.run([str(binary), str(count)], capture_output=True, text=True, check=True)
    aligned_count, mask_bytes, valid_candidates, input_bytes = (int(value) for value in completed.stdout.split())

    assert aligned_count >= count
    assert aligned_count % (256 // 4) == 0
    assert aligned_count - count < 256 // 4
    assert mask_bytes >= (aligned_count + 7) // 8
    assert input_bytes >= aligned_count * 4
    assert valid_candidates == 0
