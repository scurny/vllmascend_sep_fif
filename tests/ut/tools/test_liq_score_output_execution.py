# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
VECTOR = ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/"
    "lightning_indexer_quant_service_vector_arch35.h"
)


def _block(source: str, marker: str, start: int = 0) -> str:
    marker_start = source.index(marker, start)
    open_brace = source.index("{", marker_start)
    depth = 0
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[marker_start : index + 1]
    raise AssertionError(f"unterminated {marker}")


def test_extracted_score_writes_precede_reset_and_cover_invalid_rows(tmp_path: Path) -> None:
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("a C++ compiler is required")

    vector = VECTOR.read_text(encoding="utf-8")
    direct_start = vector.index("if (needCopyOutGm)")
    direct = _block(vector, "if (hasSparseValuesOut_)", direct_start)
    invalid_function = _block(vector, "LIQVector<LIQT>::CleanInvalidOutput")
    invalid = invalid_function[invalid_function.index("{") + 1 : -1]

    source = tmp_path / "score_output.cc"
    binary = tmp_path / "score_output"
    source.write_text(
        """
#include <cstdint>
#include <string>
#include <vector>
constexpr int BASE_TOPK = 2048;
constexpr int BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr int PIPE_V = 0;
enum class HardEvent { V_MTE3, MTE3_V };
struct Event { std::string name; int64_t offset; };
static std::vector<Event> events;
template <typename T> struct LocalTensor {
  int64_t offset = 0;
  LocalTensor operator[](int64_t delta) const { return {offset + delta}; }
  template <typename U> LocalTensor<U> ReinterpretCast() const { return {offset}; }
};
template <typename T> struct GlobalTensor {
  std::string name;
  int64_t offset = 0;
  GlobalTensor operator[](int64_t delta) const { return {name, offset + delta}; }
};
struct Buffer {
  template <typename T> LocalTensor<T> Get() { return {}; }
};
struct Queue {
  template <typename T> LocalTensor<T> AllocTensor() { return {}; }
  template <typename T> void FreeTensor(LocalTensor<T>) { events.push_back({"free", 0}); }
};
namespace LIQServiceVec {
constexpr int32_t NEG_INF = static_cast<int32_t>(0xff800000U);
template <typename T>
void CopyOut(GlobalTensor<T> dst, LocalTensor<T>, int64_t) { events.push_back({"copy_" + dst.name, dst.offset}); }
}
template <int> void PipeBarrier() { events.push_back({"barrier", 0}); }
template <HardEvent event> void SetWaitFlag(HardEvent) {
  events.push_back({event == HardEvent::V_MTE3 ? "wait_v" : "wait_mte3", 0});
}
template <typename T> void Duplicate(LocalTensor<T>, int32_t value, int64_t) {
  events.push_back({value == LIQServiceVec::NEG_INF ? "fill_neg_inf" : "fill_invalid", 0});
}
void Extract(LocalTensor<float>, LocalTensor<uint32_t>, LocalTensor<float>, int64_t) {
  events.push_back({"extract", 0});
}
void InitSortOutBuf(LocalTensor<float>, int64_t) { events.push_back({"reset", 0}); }
Buffer tmpBuf_;
Queue outQueue_;
LocalTensor<float> globalTopkUb_;
GlobalTensor<float> sparseValueOutGm_{"values"};
GlobalTensor<int32_t> indiceOutGm{"indices"};
struct { int64_t sparseCount = 4; int32_t INVALID_IDX = -1; } constInfo_;
bool hasSparseValuesOut_ = true;
template <typename Dummy> void RunDirect() {
  int64_t outputOffset = 37;
  int innerS1Idx = 0;
"""
        + direct
        + """
}
void RunInvalid() {
  int64_t invalidS1offset = 91;
"""
        + invalid
        + """
}
int Find(const std::string &name) {
  for (int i = 0; i < static_cast<int>(events.size()); ++i) if (events[i].name == name) return i;
  return -1;
}
int main() {
  RunDirect<int>();
  int values = Find("copy_values"), indices = Find("copy_indices"), reset = Find("reset");
  if (values < 0 || indices < 0 || reset < 0 || values > reset || indices > reset) return 1;
  if (events[values].offset != 37 || events[indices].offset != 37) return 2;
  events.clear();
  RunInvalid();
  values = Find("copy_values"); indices = Find("copy_indices");
  if (values < 0 || indices < 0 || values > indices) return 3;
  if (events[values].offset != 91 || events[indices].offset != 91) return 4;
  return 0;
}
""",
        encoding="utf-8",
    )
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
