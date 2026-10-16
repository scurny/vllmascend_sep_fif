from __future__ import annotations

import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OP_DEFINITION = REPOSITORY_ROOT / (
    "csrc/attention/quant_lightning_indexer_v2/op_host/"
    "quant_lightning_indexer_v2_def.cpp"
)
DEVICE_KERNEL = REPOSITORY_ROOT / (
    "csrc/attention/quant_lightning_indexer_v2/op_kernel/"
    "quant_lightning_indexer_v2.cpp"
)

HOST_TO_KERNEL_NAME = {
    "q": "query",
    "k": "key",
    "w": "weights",
    "q_descale": "queryScale",
    "k_descale": "keyScale",
    "cu_seqlens_q": "cuSeqlensQ",
    "cu_seqlens_k": "cuSeqlensK",
    "seqused_q": "sequsedQ",
    "seqused_k": "sequsedK",
    "cmp_residual_k": "cmpResidualK",
    "block_table": "blockTable",
    "output_idx_offset": "outputIdxOffset",
    "candidate_topk_index": "candidateTopkIndex",
    "sparse_indices": "sparseIndices",
    "sparse_values": "sparseValues",
    "candidate_topk_index_out": "candidateTopkIndexOut",
    "token_gva_table": "tokenGvaTable",
    "prefetch_staging_buf": "prefetchStagingBuf",
    "prefetch_slot_counter": "prefetchSlotCounter",
    "token_to_prefetch_slot": "tokenToPrefetchSlot",
}


def _defined_tensor_names(op_definition: str, kind: str) -> list[str]:
    definition_body = op_definition.split("OpAICoreConfig", maxsplit=1)[0]
    return re.findall(rf'this->{kind}\("([^"]+)"\)', definition_body)


def _kernel_pointer_names(kernel_source: str) -> list[str]:
    signature = re.search(
        r"void quant_lightning_indexer_v2\s*\((.*?)\)\s*\{",
        kernel_source,
        re.DOTALL,
    )
    assert signature is not None
    parameters = signature.group(1).split(",")
    assert all("__gm__ uint8_t *" in parameter for parameter in parameters)
    return [re.search(r"\*\s*(\w+)\s*$", parameter).group(1) for parameter in parameters]


def test_qli_v2_kernel_entry_matches_cann_tensor_abi() -> None:
    host_definition = OP_DEFINITION.read_text(encoding="utf-8")
    kernel_source = DEVICE_KERNEL.read_text(encoding="utf-8")

    host_tensors = _defined_tensor_names(host_definition, "Input")
    host_tensors += _defined_tensor_names(host_definition, "Output")
    expected_pointer_names = [HOST_TO_KERNEL_NAME.get(name, name) for name in host_tensors]
    expected_pointer_names.extend(("workspace", "tiling"))

    assert _kernel_pointer_names(kernel_source) == expected_pointer_names
