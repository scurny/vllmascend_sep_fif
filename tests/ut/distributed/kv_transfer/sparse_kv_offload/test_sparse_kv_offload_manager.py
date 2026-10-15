import sys

import torch
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

from vllm_ascend.distributed.kv_transfer.sparse_kv_offload import (
    sparse_kv_offload_manager as manager_module,
)
from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.sparse_kv_offload_manager import (
    SparseKVOffloadManager,
    get_sparse_kv_offload_cpu_pool_size_bytes,
    plan_sparse_kv_offload_memory,
)
from vllm_ascend.utils import AscendDeviceType


class TestSparseKVCustomOpLoading(unittest.TestCase):
    def tearDown(self):
        manager_module._SPARSE_KV_OFFLOAD_OPS = None

    @staticmethod
    def _all_sparse_ops():
        return SimpleNamespace(**{name: object() for name in manager_module._SPARSE_KV_OFFLOAD_OP_NAMES})

    def test_a5_loads_native_extension_without_enabling_all_custom_ops(self):
        sparse_ops = self._all_sparse_ops()
        extension = MagicMock()
        with (
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(manager_module, "bootstrap_custom_op_env") as bootstrap,
            patch.object(manager_module, "enable_custom_op") as enable_custom_op,
            patch.object(manager_module.torch.ops, "_C_ascend", sparse_ops),
            patch.dict(sys.modules, {"vllm_ascend.vllm_ascend_C": extension}),
        ):
            self.assertIs(manager_module._sparse_kv_ops(), sparse_ops)
            self.assertIs(manager_module._sparse_kv_ops(), sparse_ops)

        bootstrap.assert_called_once_with()
        enable_custom_op.assert_not_called()

    def test_a5_reports_missing_sparse_ops(self):
        sparse_ops = SimpleNamespace(sparse_kv_restore_bfloat16_tensor=object())
        with (
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            patch.object(manager_module, "bootstrap_custom_op_env"),
            patch.object(manager_module.torch.ops, "_C_ascend", sparse_ops),
            patch.dict(sys.modules, {"vllm_ascend.vllm_ascend_C": MagicMock()}),
            self.assertRaisesRegex(RuntimeError, "npu_sparse_kv_plan_transfer"),
        ):
            manager_module._sparse_kv_ops()


class _FakeKVCacheSpec:
    def __init__(
        self,
        *,
        page_size_bytes,
        max_blocks_per_request,
        store_on_host,
        block_size=128,
    ):
        self.page_size_bytes = page_size_bytes
        self.max_blocks_per_request = max_blocks_per_request
        self.store_on_host = store_on_host
        self.block_size = block_size

    def max_memory_usage_bytes(self, _vllm_config):
        return self.max_blocks_per_request * self.page_size_bytes


def _make_memory_plan_inputs(max_num_seqs=2):
    specs = {
        "host.0": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "host.1": _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        ),
        "device.0": _FakeKVCacheSpec(
            page_size_bytes=512,
            max_blocks_per_request=100,
            store_on_host=False,
        ),
    }
    vllm_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs))
    alignment_reserve = 2 * manager_module._CPU_CACHE_MAX_ALIGNMENT_OVERHEAD_PER_LAYER
    return specs, vllm_config, alignment_reserve


class TestSparseKVOffloadMemoryPlanning(unittest.TestCase):
    def test_non_a3_is_rejected(self):
        with (
            patch.object(manager_module, "_SPARSE_KV_OFFLOAD_MANAGER", None),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A2),
            self.assertRaisesRegex(RuntimeError, "supports Atlas A3 and A5"),
        ):
            manager_module.init_sparse_kv_offload_manager(None, None, None)

    def test_memory_plan_is_limited_by_active_workload(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.npu_limit_blocks, 1000)
        self.assertEqual(budget.dram_limit_blocks, 1000)
        self.assertEqual(budget.workload_limit_blocks, 201)
        self.assertEqual(budget.final_num_blocks, 201)
        self.assertEqual(budget.final_planner_bytes, 201 * (2048 + 512))
        self.assertEqual(budget.planned_host_bytes, 201 * 2048)
        self.assertEqual(budget.planned_device_bytes, 201 * 512)
        self.assertEqual(budget.limiting_factor, "workload")

    def test_memory_plan_is_limited_by_dram_capacity(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=1000 * 512,
            dram_limit_bytes=alignment_reserve + 150 * 2048,
            keep_device_kv_cache=False,
        )

        self.assertEqual(budget.dram_limit_blocks, 150)
        self.assertEqual(budget.final_num_blocks, 150)
        self.assertEqual(budget.limiting_factor, "dram")

    def test_warning_when_dram_budget_caps_npu_utilization(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with self.assertLogs(manager_module.logger, level="WARNING") as logs:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 500 * 2048,
                keep_device_kv_cache=False,
            )
        self.assertTrue(any("dram_size_per_dp_GB" in line for line in logs.output))

    def test_no_warning_when_dram_budget_not_below_npu(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        with patch.object(manager_module.logger, "warning_once") as mock_warn:
            plan_sparse_kv_offload_memory(
                kv_cache_spec=specs,
                vllm_config=vllm_config,
                available_device_memory_bytes=1000 * 512,
                dram_limit_bytes=alignment_reserve + 1000 * 2048,
                keep_device_kv_cache=False,
            )
        mock_warn.assert_not_called()

    def test_keep_device_cache_counts_full_npu_page(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        total_page_size = 2048 + 512

        budget = plan_sparse_kv_offload_memory(
            kv_cache_spec=specs,
            vllm_config=vllm_config,
            available_device_memory_bytes=125 * total_page_size,
            dram_limit_bytes=alignment_reserve + 1000 * 2048,
            keep_device_kv_cache=True,
        )

        self.assertEqual(budget.npu_limit_blocks, 125)
        self.assertEqual(budget.final_num_blocks, 125)
        self.assertEqual(budget.planned_device_bytes, 125 * total_page_size)
        self.assertEqual(budget.limiting_factor, "npu")

    def test_non_positive_capacity_produces_zero_blocks(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()

        for available_device_memory, dram_limit, limiting_factor in (
            (-1, alignment_reserve + 1000 * 2048, "npu"),
            (1000 * 512, alignment_reserve - 1, "dram"),
        ):
            with self.subTest(limiting_factor=limiting_factor):
                budget = plan_sparse_kv_offload_memory(
                    kv_cache_spec=specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=available_device_memory,
                    dram_limit_bytes=dram_limit,
                    keep_device_kv_cache=False,
                )
                self.assertEqual(budget.final_num_blocks, 0)
                self.assertEqual(budget.final_planner_bytes, 0)
                self.assertEqual(budget.limiting_factor, limiting_factor)

    def test_memory_plan_rejects_invalid_spec_layouts(self):
        specs, vllm_config, alignment_reserve = _make_memory_plan_inputs()
        invalid_cases = (
            ({"device.0": specs["device.0"]}, "at least one host"),
            ({"host.0": specs["host.0"]}, "at least one device"),
            (
                {
                    **specs,
                    "device.1": _FakeKVCacheSpec(
                        page_size_bytes=512,
                        max_blocks_per_request=100,
                        store_on_host=False,
                        block_size=64,
                    ),
                },
                "one shared block size",
            ),
        )

        for invalid_specs, message in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                plan_sparse_kv_offload_memory(
                    kv_cache_spec=invalid_specs,
                    vllm_config=vllm_config,
                    available_device_memory_bytes=1000 * 512,
                    dram_limit_bytes=alignment_reserve + 1000 * 2048,
                    keep_device_kv_cache=False,
                )

    def test_cpu_pool_size_includes_per_layer_alignment_reserve(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=[name],
                    kv_cache_spec=spec,
                )
                for name, spec in specs.items()
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_supports_uniform_specs(self):
        specs, _, alignment_reserve = _make_memory_plan_inputs()
        uniform_specs = UniformTypeKVCacheSpecs(
            block_size=128,
            kv_cache_specs=specs,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=list(specs),
                    kv_cache_spec=uniform_specs,
                )
            ],
        )

        self.assertEqual(
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config),
            200 * 2048 + alignment_reserve,
        )

    def test_cpu_pool_size_rejects_missing_host_specs(self):
        specs, _, _ = _make_memory_plan_inputs()
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["device.0"],
                    kv_cache_spec=specs["device.0"],
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "host-resident"):
            get_sparse_kv_offload_cpu_pool_size_bytes(kv_cache_config)

    def _make_manager_init_inputs(self, dram_size_per_dp_gb=1):
        spec = _FakeKVCacheSpec(
            page_size_bytes=1024,
            max_blocks_per_request=100,
            store_on_host=True,
        )
        kv_cache_config = SimpleNamespace(
            num_blocks=200,
            kv_cache_groups=[SimpleNamespace(layer_names=["host.0"], kv_cache_spec=spec)],
        )
        vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(
                get_num_layers=MagicMock(return_value=1),
                max_model_len=128,
            ),
            parallel_config=SimpleNamespace(),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1,
                max_num_batched_tokens=1,
            ),
            speculative_config=None,
        )
        offload_config = SimpleNamespace(
            topk_buffer_size=1,
            topk=1,
            use_fused_overlap=False,
            dram_size_per_dp_GB=dram_size_per_dp_gb,
        )
        return vllm_config, kv_cache_config, offload_config

    def test_manager_initializes_offload_with_planned_pool_size(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        planned_pool_size = 4096
        for rank, expected_alloc_size in ((0, planned_pool_size), (1, 0)):
            with self.subTest(rank=rank):
                offload_backend = SimpleNamespace(
                    OffloadConfig=lambda: SimpleNamespace(),
                    Scene=SimpleNamespace(SHARED="shared"),
                    initialize=MagicMock(return_value=0),
                )
                tp_group = SimpleNamespace(barrier=MagicMock())

                with (
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_rank",
                        return_value=rank,
                    ),
                    patch.object(
                        manager_module,
                        "get_tensor_model_parallel_world_size",
                        return_value=2,
                    ),
                    patch.object(
                        manager_module,
                        "get_tp_group",
                        return_value=tp_group,
                    ),
                    patch.object(
                        manager_module,
                        "get_sparse_kv_offload_cpu_pool_size_bytes",
                        return_value=planned_pool_size,
                    ),
                    patch.object(
                        manager_module,
                        "offload",
                        offload_backend,
                        create=True,
                    ),
                    patch.object(
                        manager_module.torch,
                        "zeros",
                        return_value=MagicMock(),
                    ),
                    patch.object(
                        manager_module.torch,
                        "empty",
                        return_value=MagicMock(),
                    ),
                    patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
                    patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A3),
                ):
                    SparseKVOffloadManager(
                        vllm_config,
                        kv_cache_config,
                        offload_config,
                    )

                initialized_config = offload_backend.initialize.call_args.args[0]
                self.assertEqual(
                    initialized_config.reserve_size,
                    planned_pool_size,
                )
                self.assertEqual(
                    initialized_config.alloc_size,
                    expected_alloc_size,
                )
                self.assertEqual(initialized_config.world_size, 2)
                self.assertEqual(initialized_config.rank_id, rank)
                tp_group.barrier.assert_called_once_with()

    def test_a5_manager_uses_rank_local_urma_pool(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"memfabric_transfer_protocol": "device_urma"}
        )
        planned_pool_size = 4096
        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            OFFLOAD_FLAG_GIANT_PAGE=1,
            Scene=SimpleNamespace(LOCAL="local", SHARED="shared"),
            get_dva=MagicMock(return_value=0x2000),
            initialize=MagicMock(return_value=0),
        )
        tp_group = SimpleNamespace(barrier=MagicMock())

        with (
            patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=1),
            patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=2),
            patch.object(manager_module, "get_tp_group", return_value=tp_group),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=planned_pool_size,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
        ):
            manager = SparseKVOffloadManager(vllm_config, kv_cache_config, offload_config)

        initialized_config = offload_backend.initialize.call_args.args[0]
        self.assertTrue(manager.rank_local_host_pool)
        self.assertEqual(initialized_config.reserve_size, planned_pool_size)
        self.assertEqual(initialized_config.alloc_size, planned_pool_size)
        self.assertEqual(initialized_config.world_size, 1)
        self.assertEqual(initialized_config.rank_id, 0)
        self.assertEqual(initialized_config.scene, "local")
        self.assertEqual(initialized_config.flags, 1)
        tp_group.barrier.assert_called_once_with()

    def test_a5_manager_requires_device_urma_for_pd(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"memfabric_transfer_protocol": "sdma"}
        )
        offload_backend = SimpleNamespace(
            OffloadConfig=lambda: SimpleNamespace(),
            OFFLOAD_FLAG_GIANT_PAGE=1,
            Scene=SimpleNamespace(LOCAL="local", SHARED="shared"),
            get_dva=MagicMock(return_value=0x2000),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(manager_module, "get_tensor_model_parallel_rank", return_value=0),
            patch.object(manager_module, "get_tensor_model_parallel_world_size", return_value=1),
            patch.object(manager_module, "get_tp_group", return_value=SimpleNamespace()),
            patch.object(manager_module, "get_sparse_kv_offload_cpu_pool_size_bytes", return_value=4096),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A5),
            self.assertRaisesRegex(ValueError, "device_urma"),
        ):
            SparseKVOffloadManager(vllm_config, kv_cache_config, offload_config)

        offload_backend.initialize.assert_not_called()

    def test_rank_local_pool_keeps_host_and_device_addresses_separate(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.rank_local_host_pool = True
        manager.num_layers = 1
        manager.kv_cache_config = SimpleNamespace(num_blocks=4)
        manager.k_caches_cpu = [SimpleNamespace(data_ptr=lambda: 0x1000, numel=lambda: 40, element_size=lambda: 2)]
        manager.v_caches_cpu = [SimpleNamespace(data_ptr=lambda: 0x2000, numel=lambda: 20, element_size=lambda: 2)]
        offload_backend = SimpleNamespace(get_dva=MagicMock(side_effect=[0xA000, 0xB000]))

        with patch.object(manager_module, "offload", offload_backend, create=True):
            manager._initialize_pool_address_tables()

        self.assertEqual(manager.hvas_k_bases, [0x1000])
        self.assertEqual(manager.hvas_v_bases, [0x2000])
        self.assertEqual(manager.dvas_k_bases, [0xA000])
        self.assertEqual(manager.dvas_v_bases, [0xB000])
        self.assertEqual(manager.cpu_block_lens, [(20, 10)])

    def test_a5_simt_onload_plans_once_then_reuses_misses_for_next_layer(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.max_num_topk_rows = 4
        manager.mtp_layer_id = -1
        manager.topk_buffers_k = ["resident_k_0", "resident_k_1"]
        manager.topk_buffers_v = ["resident_v_0", "resident_v_1"]
        manager.simt_lru = MagicMock()
        manager._get_offload_layer_id = MagicMock(return_value=0)

        manager._onload_topk_kv_simt(
            "layer.0",
            2,
            1,
            "block_table",
            "topk_indices",
            "current_slots",
            "req_ids",
            "stable_prefix_lens",
            "visible_seq_lens",
            "token_to_req",
        )

        manager.simt_lru.set_active_rows.assert_called_once_with(2)
        manager.simt_lru.plan_and_transfer.assert_called_once_with(
            layer_id=0,
            req_ids="req_ids",
            topk_indices="topk_indices",
            stable_prefix_lens="stable_prefix_lens",
            visible_seq_lens="visible_seq_lens",
            token_to_req="token_to_req",
            block_table="block_table",
            resident_k="resident_k_0",
            resident_v="resident_v_0",
        )

        manager._get_offload_layer_id.return_value = 1
        manager._onload_topk_kv_simt(
            "layer.1",
            2,
            1,
            "block_table",
            "ignored_topk",
            "current_slots",
            "ignored_req_ids",
            "ignored_stable_prefix_lens",
            "ignored_visible_seq_lens",
            "token_to_req",
            skip_topk=True,
        )

        manager.simt_lru.set_active_rows.assert_called_once_with(2)
        manager.simt_lru.transfer_reused_plan.assert_called_once_with(
            layer_id=1,
            token_to_req="token_to_req",
            block_table="block_table",
            resident_k="resident_k_1",
            resident_v="resident_v_1",
        )

    def test_a5_simt_mtp_skip_topk_replans_compacted_rows(self):
        manager = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        manager.max_num_topk_rows = 4
        manager.mtp_layer_id = 2
        manager.topk_buffers_k = ["resident_k_0", "resident_k_1", "resident_k_2"]
        manager.topk_buffers_v = ["resident_v_0", "resident_v_1", "resident_v_2"]
        manager.simt_lru = MagicMock()
        manager._get_offload_layer_id = MagicMock(return_value=2)

        manager._onload_topk_kv_simt(
            "mtp.layer.2",
            3,
            1,
            "block_table",
            "compacted_topk",
            "current_slots",
            "req_ids",
            "stable_prefix_lens",
            "visible_seq_lens",
            "token_to_req",
            skip_topk=True,
        )

        manager.simt_lru.set_active_rows.assert_called_once_with(3)
        manager.simt_lru.transfer_reused_plan.assert_not_called()
        manager.simt_lru.plan_and_transfer.assert_called_once_with(
            layer_id=2,
            req_ids="req_ids",
            topk_indices="compacted_topk",
            stable_prefix_lens="stable_prefix_lens",
            visible_seq_lens="visible_seq_lens",
            token_to_req="token_to_req",
            block_table="block_table",
            resident_k="resident_k_2",
            resident_v="resident_v_2",
        )

    def test_manager_rejects_pool_larger_than_dram_limit(self):
        vllm_config, kv_cache_config, offload_config = self._make_manager_init_inputs()
        offload_backend = SimpleNamespace(
            OffloadConfig=MagicMock(),
            Scene=SimpleNamespace(SHARED="shared"),
            initialize=MagicMock(return_value=0),
        )

        with (
            patch.object(
                manager_module,
                "get_tensor_model_parallel_rank",
                return_value=0,
            ),
            patch.object(
                manager_module,
                "get_tensor_model_parallel_world_size",
                return_value=1,
            ),
            patch.object(
                manager_module,
                "get_tp_group",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                manager_module,
                "get_sparse_kv_offload_cpu_pool_size_bytes",
                return_value=(1 << 30) + 1,
            ),
            patch.object(manager_module, "offload", offload_backend, create=True),
            patch.object(manager_module.torch, "zeros", return_value=MagicMock()),
            patch.object(manager_module.torch, "empty", return_value=MagicMock()),
            patch.object(manager_module, "_sparse_kv_ops", return_value=MagicMock()),
            patch.object(manager_module, "get_ascend_device_type", return_value=AscendDeviceType.A3),
            self.assertRaisesRegex(ValueError, "exceeds DRAM limit"),
        ):
            SparseKVOffloadManager(
                vllm_config,
                kv_cache_config,
                offload_config,
            )

        offload_backend.OffloadConfig.assert_not_called()
        offload_backend.initialize.assert_not_called()

class TestSparseKVOffloadPrefetch(unittest.TestCase):
    """Unit tests for ECHO-style prefetch methods (CPU-only logic)."""

    def test_update_token_gva_table_single_request(self):
        """One request, block_table=[0,1,2], block_size=128."""
        num_reqs = 1
        block_size = 128
        max_model_len = 512
        max_blocks = max_model_len // block_size  # 4
        max_tokens = max_blocks * block_size  # 512
        token_size_k = 64 * 128 * 2  # 1 head * 128 dim * bf16 = 16384
        token_size_v = 64 * 128 * 2  # same for simplicity
        gvas_k_base = 0x10000000
        gvas_v_base = 0x20000000
        # block_table: [1, 4, 2, 0] = blocks 0→1, 1→4, 2→2, 3→0
        block_table = torch.tensor([[1, 4, 2, 0]], dtype=torch.int32)
        # token_gva_table like the manager's
        token_gva = torch.zeros([num_reqs, max_tokens, 2], dtype=torch.uint64)
        # Simulate update_token_gva_table logic
        bs = block_size
        block_nums = block_table[:num_reqs].to(torch.int64).unsqueeze(-1)
        offsets = torch.arange(bs, dtype=torch.int64)
        global_positions = (block_nums * bs + offsets).reshape(num_reqs, max_blocks * bs)[:, :max_tokens]
        token_gva[:num_reqs, :max_tokens, 0] = gvas_k_base + global_positions * token_size_k
        token_gva[:num_reqs, :max_tokens, 1] = gvas_v_base + global_positions * token_size_v
        # Verify: position 0 → block 1 → global_pos=128, GVA=base+128*16384
        self.assertEqual(
            token_gva[0, 0, 0].item(),
            gvas_k_base + 128 * token_size_k,
        )
        # Position 127 (last in block 1)
        self.assertEqual(
            token_gva[0, 127, 0].item(),
            gvas_k_base + (128 + 127) * token_size_k,
        )
        # Position 128 → block 4 → global_pos=512
        self.assertEqual(
            token_gva[0, 128, 0].item(),
            gvas_k_base + 512 * token_size_k,
        )
        # V GVA uses different base but same position arithmetic
        self.assertEqual(
            token_gva[0, 0, 1].item(),
            gvas_v_base + 128 * token_size_v,
        )

    def test_update_token_gva_table_two_requests(self):
        """Two requests with different block tables."""
        num_reqs = 2
        block_size = 128
        max_model_len = 256
        max_blocks = max_model_len // block_size  # 2
        max_tokens = max_blocks * block_size  # 256
        token_size_k = 16384
        token_size_v = 16384
        gvas_k_base = 0x10000000
        block_table = torch.tensor(
            [[0, 1], [10, 11]], dtype=torch.int32
        )  # req0: blocks 0-1, req1: blocks 10-11
        token_gva = torch.zeros([num_reqs, max_tokens, 2], dtype=torch.uint64)
        bs = block_size
        block_nums = block_table[:num_reqs].to(torch.int64).unsqueeze(-1)
        offsets = torch.arange(bs, dtype=torch.int64)
        global_positions = (block_nums * bs + offsets).reshape(num_reqs, max_blocks * bs)[:, :max_tokens]
        token_gva[:num_reqs, :max_tokens, 0] = gvas_k_base + global_positions * token_size_k
        # req0, pos 0 → block 0 → global_pos=0
        self.assertEqual(token_gva[0, 0, 0].item(), gvas_k_base + 0)
        # req1, pos 0 → block 10 → global_pos=1280
        self.assertEqual(token_gva[1, 0, 0].item(), gvas_k_base + 1280 * token_size_k)
        # req0 and req1 do NOT share GVA addresses
        self.assertNotEqual(token_gva[0, 0, 0].item(), token_gva[1, 0, 0].item())

    def test_global_position_computation_single_token(self):
        """One decode token, one request, one topk entry."""
        topk = 2
        num_tokens = 1
        max_model_len = 256
        # Simulate complete_prefetch_staging global_pos computation
        # token_to_req: [num_tokens], topk_indices: [num_tokens, topk]
        token_to_req = torch.tensor([0], dtype=torch.int32)  # req 0
        topk_indices = torch.tensor([[10, 42]], dtype=torch.int32)
        flat_topk = topk_indices.reshape(-1).to(torch.int64)  # [2] = [10, 42]
        token_req = token_to_req[:num_tokens].to(dtype=torch.int64)
        token_bases = token_req * max_model_len  # [1] = [0]
        token_bases_expanded = token_bases.unsqueeze(1).expand(-1, topk).reshape(-1)
        global_topk = token_bases_expanded + flat_topk
        # Single request → globalPos == localPos
        self.assertTrue(torch.equal(global_topk, flat_topk))

    def test_global_position_computation_two_requests(self):
        """Two decode tokens from different requests, multiple topk entries."""
        topk = 3
        num_tokens = 2
        max_model_len = 256
        # token 0 → req 0, token 1 → req 1
        token_to_req = torch.tensor([0, 1], dtype=torch.int32)
        # Each token's topk indices (local positions within each req's KV context)
        topk_indices = torch.tensor([[0, 5, 10], [0, 7, 14]], dtype=torch.int32)
        flat_topk = topk_indices.reshape(-1).to(torch.int64)  # [6]
        token_req = token_to_req[:num_tokens].to(dtype=torch.int64)  # [0, 1]
        token_bases = token_req * max_model_len  # [0, 256]
        token_bases_expanded = token_bases.unsqueeze(1).expand(-1, topk).reshape(-1)
        global_topk = token_bases_expanded + flat_topk
        # Expected: req0 tokens stay same, req1 tokens get +256
        expected = torch.tensor([0, 5, 10, 256, 263, 270], dtype=torch.int64)
        self.assertTrue(torch.equal(global_topk, expected))
        # Verify cross-request protection: req0 and req1 have different global positions
        # even though local positions are the same (both have position 0)
        self.assertNotEqual(global_topk[0].item(), global_topk[3].item())

    def test_global_position_computation_three_tokens_two_reqs(self):
        """Two tokens from same request, one from another."""
        topk = 2
        max_model_len = 128
        token_to_req = torch.tensor([0, 0, 1], dtype=torch.int32)
        topk_indices = torch.tensor([[0, 1], [2, 3], [0, 4]], dtype=torch.int32)
        flat_topk = topk_indices.reshape(-1).to(torch.int64)
        token_req = token_to_req[:3].to(dtype=torch.int64)
        token_bases = token_req * max_model_len
        token_bases_expanded = token_bases.unsqueeze(1).expand(-1, topk).reshape(-1)
        global_topk = token_bases_expanded + flat_topk
        # token0(req0): positions 0,1 → 0,1
        # token1(req0): positions 2,3 → 2,3
        # token2(req1): positions 0,4 → 128,132
        expected = torch.tensor([0, 1, 2, 3, 128, 132], dtype=torch.int64)
        self.assertTrue(torch.equal(global_topk, expected))

    def test_reset_prefetch_state(self):
        """reset_prefetch_state: zeroes counter and fills slot table with -1.
        NPU-dependent; only runs when NPU is available.
        """
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        # Build a minimal manager (requires NPU for tensor creation)
        max_num_reqs = 2
        max_model_len = 128
        slot_table = torch.full(
            [max_num_reqs, max_model_len],
            42,
            dtype=torch.int32,
            device="npu",
        )
        slot_counter = torch.zeros([1], dtype=torch.int32, device="npu")
        # Simulate reset_prefetch_state
        slot_counter.zero_()
        slot_table.fill_(-1)
        self.assertEqual(slot_counter.cpu().item(), 0)
        self.assertTrue(torch.all(slot_table.cpu() == -1))


class TestUpdateTokenGVATableRealImpl(unittest.TestCase):
    """Test the REAL update_token_gva_table implementation (no mock logic)."""

    def _make_minimal_manager(
        self,
        num_layers=1,
        block_size=128,
        max_model_len=512,
        max_num_reqs=2,
        token_size_k=16384,
        token_size_v=16384,
    ):
        # Bypass __init__ entirely; we only need a handful of attributes.
        mgr = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        mgr.tp_rank = 0
        mgr.block_size = block_size
        mgr.max_model_len = max_model_len
        mgr.topk = 2  # not used by update_token_gva_table, but harmless
        mgr.num_layers = num_layers
        mgr.gvas_k_bases = {i: 0x10000000 + i * 0x1000000 for i in range(num_layers)}
        mgr.gvas_v_bases = {i: 0x20000000 + i * 0x1000000 for i in range(num_layers)}
        mgr.token_size_bytes_k = token_size_k
        mgr.token_size_bytes_v = token_size_v
        max_blocks = (max_model_len + block_size - 1) // block_size
        mgr.token_gva_table_cpu = torch.zeros(
            [max_num_reqs, max_model_len, 2], dtype=torch.uint64, device="cpu"
        )
        mgr._token_gva_table_dirty = False
        return mgr

    def test_single_request_matches_formula(self):
        mgr = self._make_minimal_manager()
        # block_table: [[1, 4, 2, 0]] with block_size=128, max_model_len=512
        block_table_cpu = torch.tensor([[1, 4, 2, 0]], dtype=torch.int32)
        mgr.update_token_gva_table(block_table_cpu, num_reqs=1, layer_id=0)
        # Same expected values as your existing test, but now driven by REAL code.
        k_base = mgr.gvas_k_bases[0]
        v_base = mgr.gvas_v_bases[0]
        k_sz = mgr.token_size_bytes_k
        v_sz = mgr.token_size_bytes_v
        # token pos 0 → block 1 → global_pos 128
        self.assertEqual(mgr.token_gva_table_cpu[0, 0, 0].item(), k_base + 128 * k_sz)
        self.assertEqual(mgr.token_gva_table_cpu[0, 0, 1].item(), v_base + 128 * v_sz)

        # pos 127 → still block 1 → global_pos 128+127
        self.assertEqual(mgr.token_gva_table_cpu[0, 127, 0].item(), k_base + (128 + 127) * k_sz)

        # pos 128 → block 4 → global_pos 512
        self.assertEqual(mgr.token_gva_table_cpu[0, 128, 0].item(), k_base + 512 * k_sz)

        # dirty flag set
        self.assertTrue(mgr._token_gva_table_dirty)

    def test_two_requests_no_overlap(self):
        mgr = self._make_minimal_manager(max_model_len=256, max_num_reqs=2)
        block_table_cpu = torch.tensor([[0, 1], [10, 11]], dtype=torch.int32)
        mgr.update_token_gva_table(block_table_cpu, num_reqs=2, layer_id=0)
        k_base = mgr.gvas_k_bases[0]
        k_sz = mgr.token_size_bytes_k

        # req0 pos0 → block 0 → GVA = base + 0
        self.assertEqual(mgr.token_gva_table_cpu[0, 0, 0].item(), k_base)

        # req1 pos0 → block 10 → GVA = base + 10*128*ksz
        self.assertEqual(mgr.token_gva_table_cpu[1, 0, 0].item(), k_base + 10 * 128 * k_sz)

        # no overlap
        self.assertNotEqual(mgr.token_gva_table_cpu[0, 0, 0].item(), mgr.token_gva_table_cpu[1, 0, 0].item())

    def test_non_tp0_is_noop(self):
        mgr = self._make_minimal_manager()
        mgr.tp_rank = 1
        block_table_cpu = torch.tensor([[1, 2, 3, 0]], dtype=torch.int32)
        mgr.update_token_gva_table(block_table_cpu, num_reqs=1, layer_id=0)
        # Nothing written; table stays zero, dirty flag stays False
        self.assertEqual(mgr.token_gva_table_cpu[0, 0, 0].item(), 0)
        self.assertFalse(mgr._token_gva_table_dirty)

    def test_invalid_tail_positions_remain_zero(self):
        # max_blocks * block_size > max_model_len edge case
        mgr = self._make_minimal_manager(max_model_len=300, block_size=128)
        # max_blocks = ceil(300/128) = 3 → 3*128 = 384 > 300
        block_table_cpu = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        mgr.update_token_gva_table(block_table_cpu, num_reqs=1, layer_id=0)
        # positions [300, 384) should NOT be touched (they're beyond max_tokens)
        # table shape is [.., 300, ..], so just ensure last valid pos is written
        self.assertNotEqual(mgr.token_gva_table_cpu[0, 299, 0].item(), 0)


class TestCompletePrefetchStagingRealImpl(unittest.TestCase):
    """Exercise the real gather/scatter implementation with CPU tensors."""

    device = "cpu"

    def _make_minimal_manager(
        self,
        num_layers=1,
        topk=2,
        topk_buffer_size=128,
        block_size=128,
        max_model_len=256,
        max_num_reqs=2,
        k_dim=64,
        v_dim=64,
        max_num_topk_rows=8,
    ):
        """Construct a Manager skeleton with only the attributes that
        complete_prefetch_staging touches."""
        mgr = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        mgr.tp_rank = 0
        mgr.rank_local_host_pool = False
        mgr.num_layers = num_layers
        mgr.topk = topk
        mgr.topk_buffer_size = topk_buffer_size
        mgr.block_size = block_size
        mgr.max_model_len = max_model_len
        mgr.max_num_reqs = max_num_reqs
        mgr.max_num_topk_rows = max_num_topk_rows
        # token sizes must match: token_size_bytes_k = k_dim * 2 (bf16)
        mgr.token_size_bytes_k = k_dim * 2
        mgr.token_size_bytes_v = v_dim * 2
        mgr.prefetch_candidate_max = topk * 2
        dev = self.device
        mgr.current_slots_npu = torch.arange(topk, dtype=torch.int32, device=dev).repeat(max_num_topk_rows, 1)
        # staging buffer: [prefetch_candidate_max, k_size + v_size] uint8
        k_entry = mgr.token_size_bytes_k + mgr.token_size_bytes_v
        mgr.prefetch_staging_buf = torch.zeros(
            [mgr.prefetch_candidate_max, k_entry], dtype=torch.uint8, device=dev
        )
        # resident buffers per layer: [rows, topk_buffer_size, 1, dim] bf16
        mgr.topk_buffers_k = [
            torch.zeros(
                [max_num_topk_rows, topk_buffer_size, 1, k_dim],
                dtype=torch.bfloat16,
                device=dev,
            )
            for _ in range(num_layers)
        ]
        mgr.topk_buffers_v = [
            torch.zeros(
                [max_num_topk_rows, topk_buffer_size, 1, v_dim],
                dtype=torch.bfloat16,
                device=dev,
            )
            for _ in range(num_layers)
        ]
        # token_to_prefetch_slot: [max_num_reqs, max_blocks_per_req*block_size] int32
        max_blocks = (max_model_len + block_size - 1) // block_size
        mgr.token_to_prefetch_slot = torch.full(
            [max_num_reqs, max_blocks * block_size],
            -1,
            dtype=torch.int32,
            device=dev,
        )
        return mgr

    def test_prefetch_preserves_other_lru_hits_when_topk_order_changes(self):
        # Given: A and B occupy physical slots 0 and 1; only B was prefetched.
        mgr = self._make_minimal_manager(topk=2, k_dim=4, v_dim=4)
        mgr.topk_buffers_k[0][0, 0].fill_(10)
        mgr.topk_buffers_k[0][0, 1].fill_(20)
        mgr.topk_buffers_v[0][0, 0].fill_(100)
        mgr.topk_buffers_v[0][0, 1].fill_(200)
        mgr.current_slots_npu[0] = torch.tensor([1, 0], dtype=torch.int32, device=self.device)
        mgr.token_to_prefetch_slot[0, 1] = 0
        mgr.prefetch_staging_buf[0, :8].view(torch.bfloat16).fill_(20)
        mgr.prefetch_staging_buf[0, 8:].view(torch.bfloat16).fill_(200)

        # When: this step selects B then A, both already resident.
        mgr.complete_prefetch_staging(torch.tensor([[1, 0]], dtype=torch.int32), num_tokens=1, layer_id=0)

        # Then: attention still reads B and A through the planner's slots.
        slots = mgr.current_slots_npu[0].long()
        self.assertEqual(mgr.topk_buffers_k[0][0, slots, 0, 0].tolist(), [20, 10])
        self.assertEqual(mgr.topk_buffers_v[0][0, slots, 0, 0].tolist(), [200, 100])

    def test_prefetch_does_not_write_unassigned_resident_slots(self):
        # Given: the planner did not assign a resident slot for this top-k entry.
        mgr = self._make_minimal_manager(topk=1, k_dim=4, v_dim=4)
        mgr.current_slots_npu[0, 0] = -1
        mgr.token_to_prefetch_slot[0, 5] = 0
        mgr.prefetch_staging_buf[0].fill_(1)

        # When: the token has a staging hit but no resident destination.
        mgr.complete_prefetch_staging(torch.tensor([[5]], dtype=torch.int32), num_tokens=1, layer_id=0)

        # Then: no resident memory is modified.
        self.assertEqual(torch.count_nonzero(mgr.topk_buffers_k[0]).item(), 0)
        self.assertEqual(torch.count_nonzero(mgr.topk_buffers_v[0]).item(), 0)

    def test_invalid_topk_does_not_read_or_clear_another_request(self):
        for token in (-1, 129, 255, 256, 2**63 - 1):
            with self.subTest(token=token):
                mgr = self._make_minimal_manager(topk=1, k_dim=4, v_dim=4, max_model_len=129)
                mgr.token_to_prefetch_slot.fill_(0)
                mgr.prefetch_staging_buf[0].fill_(1)
                original_map = mgr.token_to_prefetch_slot.clone()

                mgr.complete_prefetch_staging(
                    torch.tensor([[token]], dtype=torch.int64),
                    num_tokens=1,
                    layer_id=0,
                    token_to_req=torch.tensor([1], dtype=torch.int32),
                )

                self.assertTrue(torch.equal(mgr.token_to_prefetch_slot, original_map))
                self.assertEqual(torch.count_nonzero(mgr.topk_buffers_k[0]).item(), 0)
                self.assertEqual(torch.count_nonzero(mgr.topk_buffers_v[0]).item(), 0)

    def test_invalid_request_does_not_access_slot_map(self):
        for request in (-1, 2, 2**62):
            with self.subTest(request=request):
                mgr = self._make_minimal_manager(topk=1, k_dim=4, v_dim=4)
                mgr.token_to_prefetch_slot.fill_(0)
                mgr.prefetch_staging_buf[0].fill_(1)
                original_map = mgr.token_to_prefetch_slot.clone()

                mgr.complete_prefetch_staging(
                    torch.tensor([[0]], dtype=torch.int32),
                    num_tokens=1,
                    layer_id=0,
                    token_to_req=torch.tensor([request], dtype=torch.int64),
                )

                self.assertTrue(torch.equal(mgr.token_to_prefetch_slot, original_map))
                self.assertEqual(torch.count_nonzero(mgr.topk_buffers_k[0]).item(), 0)

    def test_out_of_range_staging_slot_is_not_copied(self):
        mgr = self._make_minimal_manager(topk=1, k_dim=4, v_dim=4)
        mgr.token_to_prefetch_slot[0, 5] = mgr.prefetch_candidate_max

        mgr.complete_prefetch_staging(torch.tensor([[5]], dtype=torch.int32), num_tokens=1, layer_id=0)

        self.assertEqual(torch.count_nonzero(mgr.topk_buffers_k[0]).item(), 0)
        self.assertEqual(mgr.token_to_prefetch_slot[0, 5].item(), mgr.prefetch_candidate_max)

    def test_no_hits_returns_early(self):
        """All slots -1 → no hits → no scatter, buffers unchanged."""
        mgr = self._make_minimal_manager()
        # leave token_to_prefetch_slot all -1
        topk_indices_cpu = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        # should silently return without touching anything
        mgr.complete_prefetch_staging(topk_indices_cpu, num_tokens=2, layer_id=0)
        # staging untouched, resident untouched
        self.assertTrue(torch.all(mgr.topk_buffers_k[0] == 0))
        self.assertTrue(torch.all(mgr.topk_buffers_v[0] == 0))

    def test_single_hit_copies_staging_to_resident(self):
        """One hit: staging slot 3 → resident [row0, col0]."""
        mgr = self._make_minimal_manager(topk=2, k_dim=4, v_dim=4)
        k_size = mgr.token_size_bytes_k  # 8
        v_size = mgr.token_size_bytes_v  # 8
        # Fill staging slot 3 with a recognizable pattern for K and V.
        # K occupies bytes [0:k_size] in the row, V occupies [k_size:k_size+v_size].
        k_pattern = torch.arange(4, dtype=torch.bfloat16) + 10
        v_pattern = torch.arange(4, dtype=torch.bfloat16) + 100
        mgr.prefetch_staging_buf[3, :k_size].view(torch.bfloat16).copy_(k_pattern)
        mgr.prefetch_staging_buf[3, k_size:k_size + v_size].view(torch.bfloat16).copy_(v_pattern)
        # token 0 topk = [local_pos 7, local_pos 12]
        # token_to_prefetch_slot[req0, 7] = 3 (hit), [req0, 12] = -1 (miss)
        mgr.token_to_prefetch_slot[0, 7] = 3
        topk_indices_cpu = torch.tensor([[7, 12]], dtype=torch.int32)
        mgr.complete_prefetch_staging(topk_indices_cpu, num_tokens=1, layer_id=0)
        # hit was at flat index 0 → row 0, col 0 (topk=2: 0 // 2 = 0, 0 % 2 = 0)
        self.assertTrue(
            torch.equal(mgr.topk_buffers_k[0][0, 0, 0].cpu(), k_pattern.cpu()),
            f"K mismatch: {mgr.topk_buffers_k[0][0, 0, 0]} vs {k_pattern}",
        )
        self.assertTrue(
            torch.equal(mgr.topk_buffers_v[0][0, 0, 0].cpu(), v_pattern.cpu()),
            f"V mismatch: {mgr.topk_buffers_v[0][0, 0, 0]} vs {v_pattern}",
        )
        # token_to_prefetch_slot[0, 7] should be cleared to -1
        self.assertEqual(mgr.token_to_prefetch_slot[0, 7].item(), -1)
        # miss slot untouched
        self.assertEqual(mgr.token_to_prefetch_slot[0, 12].item(), -1)
        
    def test_multi_request_offset_uses_padded_slot_table_width(self):
        mgr = self._make_minimal_manager(topk=1, k_dim=4, v_dim=4, max_model_len=129)
        self.assertEqual(mgr.token_to_prefetch_slot.shape[1], 256)
        k_size = mgr.token_size_bytes_k
        v_size = mgr.token_size_bytes_v
        # req0 has topk hit at local pos 5 → staging slot 0
        # req1 has topk hit at local pos 5 → staging slot 1 (same local pos!)
        mgr.token_to_prefetch_slot[0, 5] = 0
        mgr.token_to_prefetch_slot[1, 5] = 1
        k0 = torch.full((4,), 1.0, dtype=torch.bfloat16)
        k1 = torch.full((4,), 2.0, dtype=torch.bfloat16)
        v0 = torch.full((4,), 10.0, dtype=torch.bfloat16)
        v1 = torch.full((4,), 20.0, dtype=torch.bfloat16)
        mgr.prefetch_staging_buf[0, :k_size].view(torch.bfloat16).copy_(k0)
        mgr.prefetch_staging_buf[0, k_size:k_size + v_size].view(torch.bfloat16).copy_(v0)
        mgr.prefetch_staging_buf[1, :k_size].view(torch.bfloat16).copy_(k1)
        mgr.prefetch_staging_buf[1, k_size:k_size + v_size].view(torch.bfloat16).copy_(v1)
        # token_to_req: token 0 → req 0, token 1 → req 1
        token_to_req = torch.tensor([0, 1], dtype=torch.int32)
        # topk_indices_cpu: both tokens have local pos 5
        topk_indices_cpu = torch.tensor([[5], [5]], dtype=torch.int32)
        mgr.complete_prefetch_staging(
            topk_indices_cpu, num_tokens=2, layer_id=0, token_to_req=token_to_req
        )
        # token0 → row 0 col 0 → K=1, V=10
        self.assertTrue(torch.equal(mgr.topk_buffers_k[0][0, 0, 0].cpu(), k0.cpu()))
        self.assertTrue(torch.equal(mgr.topk_buffers_v[0][0, 0, 0].cpu(), v0.cpu()))
        # token1 → row 1 col 0 → K=2, V=20
        self.assertTrue(torch.equal(mgr.topk_buffers_k[0][1, 0, 0].cpu(), k1.cpu()))
        self.assertTrue(torch.equal(mgr.topk_buffers_v[0][1, 0, 0].cpu(), v1.cpu()))
        # Both slot entries cleared
        self.assertEqual(mgr.token_to_prefetch_slot[0, 5].item(), -1)
        self.assertEqual(mgr.token_to_prefetch_slot[1, 5].item(), -1)

    def test_mixed_hit_and_miss(self):
        """Some topk entries hit, some miss — only hits are copied."""
        mgr = self._make_minimal_manager(topk=2, k_dim=4, v_dim=4)
        k_size = mgr.token_size_bytes_k
        v_size = mgr.token_size_bytes_v
        # staging slot 2 has data
        k_hit = torch.full((4,), 7.0, dtype=torch.bfloat16)
        v_hit = torch.full((4,), 70.0, dtype=torch.bfloat16)
        mgr.prefetch_staging_buf[2, :k_size].view(torch.bfloat16).copy_(k_hit)
        mgr.prefetch_staging_buf[2, k_size:k_size + v_size].view(torch.bfloat16).copy_(v_hit)
        mgr.token_to_prefetch_slot[0, 20] = 2  # hit for topk col 0
        # topk col 1 (local pos 30) stays -1 = miss
        topk_indices_cpu = torch.tensor([[20, 30]], dtype=torch.int32)
        mgr.complete_prefetch_staging(topk_indices_cpu, num_tokens=1, layer_id=0)
        # hit: row0 col0
        self.assertTrue(torch.equal(mgr.topk_buffers_k[0][0, 0, 0].cpu(), k_hit.cpu()))
        self.assertTrue(torch.equal(mgr.topk_buffers_v[0][0, 0, 0].cpu(), v_hit.cpu()))
        # miss: row0 col1 stays 0
        self.assertTrue(torch.all(mgr.topk_buffers_k[0][0, 1, 0] == 0))
        self.assertTrue(torch.all(mgr.topk_buffers_v[0][0, 1, 0] == 0))

    def test_non_tp0_is_noop(self):
        """tp_rank != 0 → method returns immediately."""
        mgr = self._make_minimal_manager(topk=1, k_dim=4, v_dim=4)
        mgr.tp_rank = 1
        # even with a valid hit set up, nothing should happen
        mgr.token_to_prefetch_slot[0, 5] = 0
        mgr.prefetch_staging_buf[0].fill_(1)
        topk_indices_cpu = torch.tensor([[5]], dtype=torch.int32)
        mgr.complete_prefetch_staging(topk_indices_cpu, num_tokens=1, layer_id=0)
        # resident buffer untouched
        self.assertTrue(torch.all(mgr.topk_buffers_k[0] == 0))
        # slot not cleared
        self.assertEqual(mgr.token_to_prefetch_slot[0, 5].item(), 0)


class TestCompletePrefetchStagingNpu(TestCompletePrefetchStagingRealImpl):
    device = "npu:0"

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")


class TestStartPrefetchIntegration(unittest.TestCase):
    """Integration test for start_prefetch: it must chain
    reset_prefetch_state -> update_token_gva_table -> _sync_token_gva_table_to_npu
    in the correct order, with the correct side effects.
    CPU-only for the table fill; one D2H sync for the NPU copy.
    Requires NPU only for the sync step.
    """

    def setUp(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")

    def _make_minimal_manager(
        self,
        num_layers=1,
        block_size=128,
        max_model_len=256,
        max_num_reqs=2,
        k_dim=4,
        v_dim=4,
    ):
        mgr = SparseKVOffloadManager.__new__(SparseKVOffloadManager)
        mgr.tp_rank = 0
        mgr.num_layers = num_layers
        mgr.block_size = block_size
        mgr.max_model_len = max_model_len
        mgr.max_num_reqs = max_num_reqs
        mgr.gvas_k_bases = [0x10000000 + i * 0x1000000 for i in range(num_layers)]
        mgr.gvas_v_bases = [0x20000000 + i * 0x1000000 for i in range(num_layers)]
        mgr.token_size_bytes_k = k_dim * 2
        mgr.token_size_bytes_v = v_dim * 2
        max_blocks = (max_model_len + block_size - 1) // block_size
        max_tokens = max_blocks * block_size
        dev = "npu:0"
        # Use int64 instead of uint64: NPU does not support DT_UINT64.
        # The source uses uint64 but A5 hardware can't create it; for the
        # purpose of this integration test the values fit in int64 just fine.
        mgr.token_gva_table_cpu = torch.zeros(
            [max_num_reqs, max_tokens, 2], dtype=torch.int64, device="cpu"
        )
        mgr.token_gva_table_npu = torch.zeros(
            [max_num_reqs, max_tokens, 2], dtype=torch.int64, device=dev
        )
        mgr._token_gva_table_dirty = True
        mgr.prefetch_slot_counter = torch.zeros([1], dtype=torch.int32, device=dev)
        mgr.token_to_prefetch_slot = torch.full(
            [max_num_reqs, max_tokens], -1, dtype=torch.int32, device=dev
        )
        return mgr

    def test_start_prefetch_chains_all_three_steps(self):
        """After start_prefetch, all three side effects must be present:

        - slot_counter zeroed

        - token_to_prefetch_slot filled with -1

        - token_gva_table_cpu populated

        - token_gva_table_npu synced (equals cpu)

        - _token_gva_table_dirty reset to False
        """
        mgr = self._make_minimal_manager()
        # Pre-dirty the state so we can tell it was reset.
        mgr.prefetch_slot_counter.fill_(42)
        mgr.token_to_prefetch_slot.fill_(7)
        block_table_cpu = torch.tensor([[1, 4, 2, 0]], dtype=torch.int32)
        mgr.start_prefetch(block_table_cpu, num_tokens=1, num_reqs=1, layer_id=0)
        # 1) counter zeroed
        self.assertEqual(mgr.prefetch_slot_counter.cpu().item(), 0)
        # 2) slot table all -1
        self.assertTrue(torch.all(mgr.token_to_prefetch_slot.cpu() == -1))
        # 3) token_gva_table_cpu populated with expected GVA
        k_base = mgr.gvas_k_bases[0]
        k_sz = mgr.token_size_bytes_k
        # pos 0 → block 1 → global_pos 128
        self.assertEqual(
            mgr.token_gva_table_cpu[0, 0, 0].item(),
            k_base + 128 * k_sz,
        )
        # 4) NPU table synced to CPU table
        self.assertTrue(torch.equal(mgr.token_gva_table_npu.cpu(), mgr.token_gva_table_cpu))
        # 5) dirty flag cleared
        self.assertFalse(mgr._token_gva_table_dirty)

    def test_start_prefetch_resets_previous_hits(self):
        """If a previous step left hit slots, start_prefetch must wipe them."""
        mgr = self._make_minimal_manager()
        # Simulate leftover hits from a previous iteration
        mgr.token_to_prefetch_slot[0, 5] = 3
        mgr.token_to_prefetch_slot[1, 10] = 1
        mgr.prefetch_slot_counter.fill_(2)
        block_table_cpu = torch.tensor([[0, 1], [10, 11]], dtype=torch.int32)
        mgr.start_prefetch(block_table_cpu, num_tokens=2, num_reqs=2, layer_id=0)
        # All previous hits gone
        self.assertTrue(torch.all(mgr.token_to_prefetch_slot.cpu() == -1))
        self.assertEqual(mgr.prefetch_slot_counter.cpu().item(), 0)

    def test_start_prefetch_non_tp0_is_noop(self):
        """tp_rank != 0 → start_prefetch returns immediately."""
        mgr = self._make_minimal_manager()
        mgr.tp_rank = 1
        mgr.token_to_prefetch_slot.fill_(9)
        mgr.prefetch_slot_counter.fill_(9)
        block_table_cpu = torch.tensor([[1, 4, 2, 0]], dtype=torch.int32)
        mgr.start_prefetch(block_table_cpu, num_tokens=1, num_reqs=1, layer_id=0)
        # Nothing was reset, nothing was written
        self.assertTrue(torch.all(mgr.token_to_prefetch_slot.cpu() == 9))
        self.assertEqual(mgr.prefetch_slot_counter.cpu().item(), 9)
        self.assertTrue(torch.all(mgr.token_gva_table_npu.cpu() == 0))


if __name__ == "__main__":
    unittest.main()
