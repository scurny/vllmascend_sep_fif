# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
PREDICTOR = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/ema_predictor.py"


def predictor(topk=2):
    spec = importlib.util.spec_from_file_location("liq_ema_predictor", PREDICTOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EMATopkPredictor(topk=topk, alpha=0.5)


def test_ema_uses_configured_topk_and_ignores_padding():
    ema = predictor()
    scores = torch.tensor([[[8.0, 4.0, 1.0]], [[2.0, -torch.inf, -torch.inf]]])
    assert ema.update(scores) == 2.0
    assert ema.update(torch.tensor([[[8.0, 6.0, 1.0]]])) == 4.0


def test_empty_or_padding_observations_preserve_threshold():
    ema = predictor()
    assert ema.update(torch.full((2, 1, 2048), -torch.inf)) == 0.0
    assert ema.update(torch.tensor([[4.0, 2.0]])) == 2.0
    assert ema.update(torch.empty(0, 1, 2048)) == 2.0
    assert ema.update(torch.full((2, 1, 2048), -torch.inf)) == 2.0


def test_nan_is_rejected_but_zero_remains_a_valid_observation():
    ema = predictor()
    assert ema.update(torch.zeros(1, 2)) == 0.0
    with pytest.raises(ValueError, match="NaN"):
        ema.update(torch.tensor([[1.0, torch.nan]]))
    assert ema.get_threshold() == 0.0


def load_methods(path, class_name, names, namespace, base="object"):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {method.name for method in methods} == set(names)
    cls.body = methods
    cls.bases = [ast.Name(id=base, ctx=ast.Load())]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(ROOT / path), "exec"), namespace)
    return namespace[class_name]


@pytest.fixture(params=["BaseDeviceAdaptor", "A5DeviceAdaptor"])
def loop(monkeypatch, request):
    thresholds = []
    score_requests = []
    manager = SimpleNamespace(
        tp_rank=0, rank_local_host_pool=True,
        token_gva_table_npu=torch.zeros(1), prefetch_staging_buf=torch.zeros(1),
        prefetch_slot_counter=torch.zeros(1), token_to_prefetch_slot=torch.zeros(1), token_size_bytes_k=128,
    )
    base_adaptor = load_methods(
        "vllm_ascend/device/device_op.py", "BaseDeviceAdaptor", ["indexer_select_post_process"],
        {"torch": torch, "get_sparse_kv_offload_manager": lambda: manager},
    )

    def quant_op(**kwargs):
        thresholds.append(kwargs.get("ema_threshold", 0.0))
        scores = kwargs.get("sparse_values_out")
        score_requests.append(scores is not None)
        if scores is not None:
            scores.fill_(-torch.inf)
            scores[0, 0, :2] = torch.tensor([8.0, 4.0])
            # A padded graph row must never lower the real decode threshold.
            scores[1, 0, :2] = torch.tensor([0.0, 0.0])
        return torch.zeros((2, 1, 2048), dtype=torch.int32)

    monkeypatch.setattr(torch.ops._C_ascend, "npu_lightning_indexer_quant", quant_op, raising=False)
    adaptor = base_adaptor
    if request.param == "A5DeviceAdaptor":
        adaptor = load_methods(
            "vllm_ascend/device/device_op.py", "A5DeviceAdaptor", ["indexer_select_post_process"],
            {"torch": torch, "BaseDeviceAdaptor": base_adaptor,
             "torch_npu": SimpleNamespace(npu_quant_lightning_indexer=quant_op)}, base="BaseDeviceAdaptor",
        )

    class Base:
        def forward(self, layer_name, hidden_states, kv_cache, metadata, output):
            if metadata is None:
                return hidden_states
            if self.fail:
                raise RuntimeError("injected failure")
            if not self.skip_topk:
                _, self.indexer._last_topk_values = adaptor.indexer_select_post_process(
                    torch.zeros(2, 64, 128, dtype=self.indexer.c8_k_cache_dtype), torch.ones(2, 64), (2, 64, 128),
                    torch.ones(2, 64), (torch.zeros(1, 128, 1, 128), torch.ones(1, 128, 1, 1)),
                    0, 1, SimpleNamespace(block_table=torch.zeros(1)), torch.tensor([1]), torch.tensor([4]),
                    True, False, ema_threshold=self.indexer._injected_ema_threshold or 0.0,
                    collect_topk_values=self.indexer._collect_prefetch_scores,
                )
            self._update_prefetch_threshold(metadata.num_decode_tokens)
            return hidden_states

    impl = load_methods(
        "vllm_ascend/attention/sfa_kv_offload.py", "AscendSFAKVOffloadImpl",
        ["forward", "_update_prefetch_threshold"], {"Base": Base, "torch": torch}, base="Base",
    )
    instance = impl()
    instance.manager = manager
    instance.indexer = SimpleNamespace(
        _last_topk_values=torch.ones(1), _injected_ema_threshold=99.0, c8_k_cache_dtype=torch.int8,
    )
    instance.ema_predictor = predictor()
    instance._next_ema_threshold = None
    instance._prefetch_enabled = True
    instance.enable_sparse_li_c8 = True
    instance.use_fused_overlap = False
    instance.skip_topk = False
    instance.fail = False
    instance.graph = False
    instance.decode = True
    instance._in_graph_runtime = lambda: instance.graph
    instance._is_decode_only = lambda metadata: metadata.num_decode_tokens > 0 and instance.decode
    metadata = SimpleNamespace(num_decode_tokens=1)
    return instance, metadata, thresholds, score_requests


def invoke(loop):
    instance, metadata, _, _ = loop
    instance.forward("layer", torch.zeros(1), (), metadata)


def test_real_scores_feed_the_immediately_following_decode(loop):
    instance, _, thresholds, requested = loop
    invoke(loop)
    invoke(loop)
    assert thresholds == [0.0, 4.0]
    assert requested == [True, True]
    assert instance.indexer._last_topk_values is None
    assert instance.indexer._injected_ema_threshold is None
    assert not instance.indexer._collect_prefetch_scores


@pytest.mark.parametrize("disabled", ["graph", "use_fused_overlap", "skip_topk", "prefill", "fp8"])
def test_ineligible_forward_does_not_collect_or_reuse_scores(loop, disabled):
    instance, _, thresholds, requested = loop
    instance._next_ema_threshold = 8.0
    if disabled == "prefill":
        instance.decode = False
    elif disabled == "fp8":
        instance.indexer.c8_k_cache_dtype = torch.float8_e4m3fn
    else:
        setattr(instance, disabled, True)
    invoke(loop)
    assert instance._next_ema_threshold == 8.0
    assert all(threshold == 0.0 for threshold in thresholds)
    assert not any(requested)
    assert instance.indexer._last_topk_values is None


def test_failed_forward_clears_injected_state(loop):
    instance, _, _, _ = loop
    instance.fail = True
    with pytest.raises(RuntimeError, match="injected failure"):
        invoke(loop)
    assert instance.indexer._last_topk_values is None
    assert instance.indexer._injected_ema_threshold is None
    assert not instance.indexer._collect_prefetch_scores


def test_profiling_without_metadata_does_not_start_prefetch(loop):
    instance, _, thresholds, requested = loop
    instance.forward("layer", torch.zeros(1), (), None)
    assert not thresholds
    assert not requested
    assert instance.indexer._injected_ema_threshold is None
    assert not instance._prefetch_active


def test_rank_without_host_pool_never_passes_uninitialized_gva(loop):
    instance, _, thresholds, _ = loop
    instance.manager.tp_rank = 1
    instance.manager.rank_local_host_pool = False
    instance._next_ema_threshold = 4.0
    invoke(loop)
    assert thresholds == [0.0]
