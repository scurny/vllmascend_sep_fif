# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / "csrc/attention/lightning_indexer_quant/lightning_indexer_quant_torch_adpt.h"
DEFINITION = ROOT / "csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_def.cpp"
KERNEL = ROOT / "csrc/attention/lightning_indexer_quant/op_kernel/lightning_indexer_quant.cpp"
META = ROOT / "csrc/torch_binding_meta.cpp"
KERNEL_HEADER = ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/lightning_indexer_quant_kernel_arch35.h"
)
TILING_HEADER = ROOT / "csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_tiling.h"
TILING = ROOT / "csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_tiling.cpp"
VECTOR = ROOT / (
    "csrc/attention/lightning_indexer_quant/op_kernel/arch35/"
    "lightning_indexer_quant_service_vector_arch35.h"
)


def _torch_schema() -> torch.FunctionSchema:
    source = (ROOT / "csrc/torch_binding.cpp").read_text(encoding="utf-8")
    declaration = re.search(
        r'ops\.def\(\s*("npu_lightning_indexer_quant\(.*?)(?=\s*\);)',
        source,
        re.DOTALL,
    )
    assert declaration is not None
    schema_text = "".join(
        ast.literal_eval(part) for part in re.findall(r'"[^"\n]*"', declaration[1])
    )
    return torch._C.parse_schema(schema_text)


@pytest.mark.parametrize("current_abi", [True, False])
def test_runner_rejects_stale_extension_before_launch(current_abi: bool) -> None:
    script = (ROOT / "tools/run_liq_test.sh").read_text()
    code = script.split('run_logged extension "$python_bin" - <<\'PY\'\n', 1)[1].split("\nPY", 1)[0]
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    arguments = _torch_schema().arguments
    if not current_abi:
        arguments = [argument for argument in arguments if argument.name != "sparse_values_out"]
    operator = SimpleNamespace(default=SimpleNamespace(_schema=SimpleNamespace(arguments=arguments)))
    namespace = {
        "Path": Path,
        "torch": SimpleNamespace(ops=SimpleNamespace(_C_ascend=SimpleNamespace(npu_lightning_indexer_quant=operator))),
        "vllm_ascend_C": SimpleNamespace(__file__=str(Path.cwd() / "vllm_ascend" / "vllm_ascend_C.so")),
    }
    if current_abi:
        exec(compile(tree, "run_liq_test.sh:extension", "exec"), namespace)
    else:
        with pytest.raises(SystemExit, match="ABI is stale; rerun with --build"):
            exec(compile(tree, "run_liq_test.sh:extension", "exec"), namespace)


def _function(source: str, name: str) -> str:
    start = source.index(name)
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


def _parameter_names(source: str, function_name: str) -> list[str]:
    start = source.index(function_name)
    open_paren = source.index("(", start)
    close_paren = source.index(")", open_paren)
    parameters = re.sub(r"//[^\n]*", "", source[open_paren + 1 : close_paren])
    return [
        match.group(1)
        for parameter in parameters.split(",")
        if (match := re.search(r"([A-Za-z_]\w*)\s*$", parameter.strip())) is not None
    ]


def test_sparse_values_buffer_is_optional_mutable_float32_input() -> None:
    schema = _torch_schema()
    argument = next(item for item in schema.arguments if item.name == "sparse_values_out")

    assert schema.arguments[-1].name == argument.name
    assert argument.default_value is None
    assert argument.alias_info is not None
    assert argument.alias_info.is_write

    declarations = re.findall(
        r'Input\("sparse_values_out"\)(.*?);', DEFINITION.read_text(encoding="utf-8"), re.DOTALL
    )
    assert len(declarations) == 2
    assert all("DT_FLOAT" in declaration for declaration in declarations)
    assert all("AutoContiguous" not in declaration for declaration in declarations)


def test_runtime_and_meta_kernels_have_the_same_parameter_order() -> None:
    adapter_parameters = _parameter_names(ADAPTER.read_text(encoding="utf-8"), "npu_lightning_indexer_quant")
    meta_parameters = _parameter_names(META.read_text(encoding="utf-8"), "npu_lightning_indexer_quant_meta")

    assert adapter_parameters == meta_parameters
    assert adapter_parameters[-1] == "sparse_values_out"


def test_adapter_validates_score_buffer_and_passes_it_before_attributes() -> None:
    adapter = ADAPTER.read_text(encoding="utf-8")

    for check in (
        'sparse_values_out->device() == query.device()',
        'sparse_values_out->scalar_type() == at::kFloat',
        'sparse_values_out->is_contiguous()',
        'sparse_values_out->sizes() == at::IntArrayRef(output_size)',
    ):
        assert check in adapter

    execute = adapter[adapter.index("EXEC_NPU_CMD(aclnnLightningIndexerQuant") :]
    assert execute.index("sparse_values_out") < execute.index("query_quant_mode")

    tiling_header = TILING_HEADER.read_text(encoding="utf-8")
    tiling = TILING.read_text(encoding="utf-8")
    assert "SPARSE_VALUES_OUT_INDEX = 12" in tiling_header
    assert "GetOptionalInputTensor(SPARSE_VALUES_OUT_INDEX)" in tiling
    assert "valuesShape.GetDimNum() != indicesShape.GetDimNum()" in tiling
    assert "valuesShape.GetDim(dim) != indicesShape.GetDim(dim)" in tiling


def test_kernel_exports_scores_on_direct_ld_and_invalid_paths() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    kernel_header = KERNEL_HEADER.read_text(encoding="utf-8")
    vector = VECTOR.read_text(encoding="utf-8")

    signature_start = kernel.index("void lightning_indexer_quant")
    signature = kernel[signature_start : kernel.index(")", signature_start)]
    assert signature.index("sparseValuesOut") < signature.index("sparseIndices")
    assert "sparseValuesOut, sparseIndices, user" in kernel
    assert "sparseValueOutBitsGm[baseSize]" in kernel_header
    assert "InitGlobalMemory(values, dealSize, LIQServiceVec::NEG_INF)" in kernel_header

    direct = vector[vector.index("if (needCopyOutGm)") : vector.index("else if (needCopyWsGm)")]
    assert "Extract(outValueUb, outIdxUb" in direct
    assert "CopyOut(sparseValueOutGm_[outputOffset], outValueUb" in direct
    assert direct.index("CopyOut(sparseValueOutGm_") < direct.index("InitSortOutBuf(")

    invalid = _function(vector, "LIQVector<LIQT>::CleanInvalidOutput")
    assert "Duplicate(idxULocal1, LIQServiceVec::NEG_INF" in invalid
    assert "CopyOut(sparseValueOutGm_[invalidS1offset], valueULocal" in invalid

    ld = _function(vector, "LIQVector<LIQT>::ProcessLD")
    assert "DataCopyPad(sparseValueOutGm_[outOffset], outValueUb" in ld
    assert "DataCopyPad(indiceOutGm[outOffset], idxULocal1" in ld
