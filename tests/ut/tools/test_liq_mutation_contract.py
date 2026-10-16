# SPDX-License-Identifier: Apache-2.0
import ast
import re
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("argument", ["prefetch_staging_buf", "token_to_prefetch_slot"])
def test_written_prefetch_tensors_have_mutable_schema(argument: str) -> None:
    source = (ROOT / "csrc/torch_binding.cpp").read_text()
    declaration = re.search(
        r'ops\.def\(\s*("npu_lightning_indexer_quant\(.*?)(?=\s*\);)', source, re.DOTALL,
    )
    assert declaration is not None
    schema_text = "".join(ast.literal_eval(part) for part in re.findall(r'"[^"\n]*"', declaration[1]))
    schema = torch._C.parse_schema(schema_text)
    parameter = next(item for item in schema.arguments if item.name == argument)
    assert parameter.alias_info is not None, f"{argument} mutation is missing from the actual schema"
    assert parameter.alias_info.is_write


@pytest.mark.parametrize("argument", ["prefetch_staging_buf", "token_to_prefetch_slot"])
def test_cann_writable_inputs_do_not_request_temporary_contiguous_storage(argument: str) -> None:
    source = (ROOT / "csrc/attention/lightning_indexer_quant/op_host/lightning_indexer_quant_def.cpp").read_text()
    declarations = re.findall(rf'Input\("{argument}"\)(.*?);', source, re.DOTALL)
    assert len(declarations) == 2
    assert all("AutoContiguous" not in declaration for declaration in declarations)
