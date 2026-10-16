from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
KERNEL_SOURCE = REPOSITORY_ROOT / "csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant.cpp"
KERNEL_HEADER = REPOSITORY_ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/lightning_indexer_quant_kernel_arch35.h"
)


def _extract_balanced_block(source: str, start: int) -> str:
    open_brace = source.index("{", start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError("unterminated C++ block")


def _invoke_macro(source: str) -> str:
    start = source.index("#define INVOKE_LI_NO_KFC_OP_IMPL")
    end = source.index("    } while (0)", start) + len("    } while (0)")
    return source[start:end]


def _process_body(source: str) -> str:
    start = source.index("template <typename LIQT>\n__aicore__ inline void LIQPreload<LIQT>::Process()")
    process = _extract_balanced_block(source, start)
    process = process.replace("template <typename LIQT>\n", "", 1)
    return process.replace("LIQPreload<LIQT>::Process()", "Preload::Process()", 1)


@pytest.fixture(scope="module")
def negative_ema_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required to execute LIQ negative EMA control flow")

    invoke_macro = _invoke_macro(KERNEL_SOURCE.read_text(encoding="utf-8"))
    process_body = _process_body(KERNEL_HEADER.read_text(encoding="utf-8"))
    source = tmp_path_factory.mktemp("liq_negative_ema") / "negative_ema.cc"
    binary = source.with_suffix("")
    source.write_text(
        "#include <cstdint>\n"
        "enum class LI_LAYOUT { TND, PA_BSND };\n"
        "namespace LIQCommon {\n"
        "constexpr float LIQ_DEBUG_STAGE_ENTRY_ONLY = -1.0f;\n"
        "constexpr float LIQ_DEBUG_STAGE_AFTER_INIT = -2.0f;\n"
        "constexpr float LIQ_DEBUG_STAGE_AFTER_MAIN = -3.0f;\n"
        "}\n"
        "template <typename, typename, typename, auto, auto, auto> struct LIQType {};\n"
        "struct Tiling { float emaThreshold; };\n"
        "using LIQTilingData = Tiling;\n"
        "struct TPipe {};\n"
        "static int init_calls;\n"
        "static int main_calls;\n"
        "static int decode_calls;\n"
        "static int invalid_calls;\n"
        "struct Preload {\n"
        "  struct { float emaThreshold; } constInfo;\n"
        "  unsigned usedCoreNum;\n"
        "  template <typename... Tail>\n"
        "  void Init(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *,\n"
        "            const LIQTilingData *tiling, TPipe *, Tail&&...) {\n"
        "    ++init_calls;\n"
        "    constInfo.emaThreshold = tiling->emaThreshold;\n"
        "    usedCoreNum = 1;\n"
        "  }\n"
        "  void ProcessMain() { ++main_calls; }\n"
        "  void ProcessDecode() { ++decode_calls; }\n"
        "  void ProcessInvalid() { ++invalid_calls; }\n"
        "  void Process();\n"
        "};\n"
        "template <typename> struct InvokePreload : Preload {};\n"
        "#define __aicore__\n"
        "#define GET_TILING_DATA_WITH_STRUCT(type, destination, source) type destination { source };\n"
        "#define LIQ_PREFETCH_ENTRY_DIAGNOSTICS(...) do { } while (0)\n"
        + invoke_macro
        + "\n"
        + process_body
        + "\n"
        "void Invoke(float threshold) {\n"
        "  Tiling tiling_data_in { threshold };\n"
        "  TPipe tPipe;\n"
        "  void *query = nullptr, *key = nullptr, *weights = nullptr, *queryScale = nullptr, *keyScale = nullptr;\n"
        "  void *actualSeqLengthsQ = nullptr, *actualSeqLengthsK = nullptr, *blocktable = nullptr;\n"
        "  void *sparseIndices = nullptr, *user = nullptr, *tokenGvaTable = nullptr, *prefetchStagingBuf = nullptr;\n"
        "  void *prefetchSlotCounter = nullptr, *tokenToPrefetchSlot = nullptr;\n"
        "  const Tiling tiling { threshold };\n"
        "  INVOKE_LI_NO_KFC_OP_IMPL(InvokePreload, int8_t, int8_t, int32_t, false,\n"
        "                            LI_LAYOUT::TND, LI_LAYOUT::PA_BSND);\n"
        "}\n"
        "int main(int argc, char **argv) {\n"
        "  const float threshold = argv[1][0] == 'm' ? -1.0f : argv[1][0] == 'n' ? -2.0f\n"
        "      : argv[1][0] == 'p' ? -3.0f : 0.0f;\n"
        "  const bool empty = argc > 2;\n"
        "  init_calls = main_calls = decode_calls = invalid_calls = 0;\n"
        "  if (empty) { Preload op {}; op.constInfo.emaThreshold = threshold; op.usedCoreNum = 0; op.Process(); }\n"
        "  else { Invoke(threshold); }\n"
        "  return empty ? !(invalid_calls == 1 && main_calls == 0 && decode_calls == 0)\n"
        "               : !(init_calls == 1 && main_calls == 1 && decode_calls == 1 && invalid_calls == 0);\n"
        "}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize("threshold", ["m", "n", "p", "z"])
def test_non_positive_ema_runs_complete_compute_path(negative_ema_binary: Path, threshold: str) -> None:
    completed = subprocess.run([str(negative_ema_binary), threshold], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_zero_core_still_uses_invalid_output_path(negative_ema_binary: Path) -> None:
    completed = subprocess.run([str(negative_ema_binary), "m", "empty"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
