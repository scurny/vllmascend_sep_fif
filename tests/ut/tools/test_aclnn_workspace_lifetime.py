# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[3]
HEADER = ROOT / "csrc/aclnn_torch_adapter/op_api_common.h"
FIXTURE = Path(__file__).parent / "fixtures/aclnn_workspace_lifetime.cc"
MARKER = "#define EXEC_NPU_CMD_MACRO"


def _read_exec_npu_cmd() -> str:
    lines = HEADER.read_text().splitlines()
    for start, line in enumerate(lines):
        if line.lstrip().startswith("#define EXEC_NPU_CMD("):
            macro_lines = [line]
            for candidate in lines[start + 1 :]:
                macro_lines.append(candidate)
                if candidate.rstrip().endswith("} while (false)"):
                    return "\n".join(macro_lines)
            break
    raise AssertionError("EXEC_NPU_CMD macro was not found or was not terminated")


def test_workspace_owner_survives_until_deferred_aclnn_call(tmp_path: Path) -> None:
    compiler = shutil.which("clang++")
    if compiler is None:
        pytest.skip("clang++ is required for the CPU-only macro contract test")

    source = FIXTURE.read_text().replace(MARKER, _read_exec_npu_cmd())
    source_path = tmp_path / "workspace_lifetime.cc"
    binary_path = tmp_path / "workspace_lifetime"
    source_path.write_text(source)

    compile_result = subprocess.run(
        [compiler, "-std=c++17", str(source_path), "-o", str(binary_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        [str(binary_path)], check=False, capture_output=True, text=True
    )
    assert run_result.returncode == 0, run_result.stderr
