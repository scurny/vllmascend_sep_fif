from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LIMITS_HEADER = (
    "csrc/attention/lightning_indexer_quant/op_host/"
    "lightning_indexer_quant_tiling_limits.h"
)


@pytest.fixture(scope="module")
def limits_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to execute LIQ tiling limits")

    source = tmp_path_factory.mktemp("liq_tiling_limits") / "limits.cc"
    binary = source.with_suffix("")
    source.write_text(
        "#include <cstdint>\n"
        "#include <cstdlib>\n"
        "#include <iostream>\n"
        "#include \"csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_tiling_limits.h\"\n"
        "int main(int argc, char **argv) {\n"
        "    if (argv[1][0] == 'r') {\n"
        "        std::cout << optiling::LIQ_PREFETCH_REQUIRED_UB_BYTES;\n"
        "        return 0;\n"
        "    }\n"
        "    const auto value = static_cast<uint64_t>(std::strtoull(argv[2], nullptr, 10));\n"
        "    const bool result = argv[1][0] == 'h'\n"
        "        ? optiling::IsSupportedLiqQueryHeadCount(static_cast<int64_t>(value))\n"
        "        : argv[1][0] == 'b' ? optiling::HasLiqBaseUbCapacity(value)\n"
        "        : optiling::HasLiqPrefetchUbCapacity(value, std::strtoull(argv[3], nullptr, 10));\n"
        "    std::cout << result;\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", "-I", str(REPOSITORY_ROOT), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


def run_limit(binary: Path, *arguments: int | str) -> bool:
    completed = subprocess.run(
        [str(binary), *(str(argument) for argument in arguments)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout == "1"


def prefetch_required_ub_bytes(binary: Path) -> int:
    completed = subprocess.run(
        [str(binary), "required"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout)


@pytest.mark.parametrize("heads", [0, 16, 32, 48, 65, 128, 2**32 + 64])
def test_liq_query_heads_reject_unsupported_raw_shape_values(limits_binary: Path, heads: int) -> None:
    assert not run_limit(limits_binary, "heads", heads)


def test_liq_query_heads_accept_exactly_64(limits_binary: Path) -> None:
    assert run_limit(limits_binary, "heads", 64)


def test_liq_prefetch_ub_rejects_small_entries_and_handles_near_uint32_entry(limits_binary: Path) -> None:
    required_ub_bytes = prefetch_required_ub_bytes(limits_binary)
    assert not run_limit(limits_binary, "ub", 0, 384)
    assert not run_limit(limits_binary, "ub", required_ub_bytes + 384 - 1, 384)
    assert run_limit(limits_binary, "ub", 2**64 - 1, 2**32 - 1)


def test_liq_prefetch_ub_accepts_exact_boundary_and_rejects_next_entry(limits_binary: Path) -> None:
    exact_boundary = prefetch_required_ub_bytes(limits_binary) + 384
    assert run_limit(limits_binary, "ub", exact_boundary, 384)
    assert not run_limit(limits_binary, "ub", exact_boundary, 416)
    large_entry_boundary = prefetch_required_ub_bytes(limits_binary) + 131_072
    assert run_limit(limits_binary, "ub", large_entry_boundary, 131_072)


def test_liq_base_ub_capacity(limits_binary: Path) -> None:
    assert not run_limit(limits_binary, "base", 0)
    assert not run_limit(limits_binary, "base", 147584 + 8192 - 1)
    assert run_limit(limits_binary, "base", 147584 + 8192)


@pytest.mark.parametrize("ub_size", [192 * 1024, 248 * 1024, 256 * 1024])
def test_real_ub_capacities_reject_oversized_rows(limits_binary: Path, ub_size: int) -> None:
    assert run_limit(limits_binary, "ub", ub_size, 384)
    assert not run_limit(limits_binary, "ub", ub_size, 131072)
    assert not run_limit(limits_binary, "ub", ub_size, 2**32 - 1)
    boundary = ub_size - 156032
    assert run_limit(limits_binary, "ub", ub_size, boundary)
    assert not run_limit(limits_binary, "ub", ub_size, boundary + 32)


def test_uint64_addition_cannot_wrap_the_budget(limits_binary: Path) -> None:
    assert not run_limit(limits_binary, "ub", 2**64 - 1, 2**64 - 1)
