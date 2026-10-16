import sys
import types

import pytest
import torch

from tools.liq_test_memory import make_source_pools


def test_hbm_pool_addresses_and_payloads_use_cpu_device():
    pools = make_source_pools(4, 8, 4, ddr=False, device="cpu")

    assert pools.k.shape == (4, 8)
    assert pools.v.shape == (4, 4)
    tokens = torch.tensor([[0, 3], [1, 2]], dtype=torch.int64)
    gva = pools.gva(tokens)
    assert gva.shape == (2, 2, 2)
    assert torch.equal(gva[..., 0], pools.k_address + tokens * 8)
    assert torch.equal(gva[..., 1], pools.v_address + tokens * 4)


def test_gva_rejects_non_integer_tokens():
    pools = make_source_pools(2, 8, 4, ddr=False, device="cpu")
    with pytest.raises(TypeError, match="integer"):
        pools.gva(torch.tensor([0.0]))


@pytest.mark.parametrize("tokens", [torch.tensor([-1]), torch.tensor([2])])
def test_gva_rejects_out_of_range_tokens(tokens):
    pools = make_source_pools(2, 8, 4, ddr=False, device="cpu")
    with pytest.raises(IndexError, match="out-of-range"):
        pools.gva(tokens)


class _FakeConfig:
    def __init__(self):
        self.device_id = None
        self.reserve_size = None
        self.alloc_size = None
        self.world_size = None
        self.rank_id = None
        self.scene = None
        self.flags = None


class _FakeOffload:
    OFFLOAD_FLAG_GIANT_PAGE = 32
    Scene = types.SimpleNamespace(LOCAL="local")
    OffloadConfig = _FakeConfig

    def __init__(self, *, init_result=0, dva=0x40000000, zero_dva=False):
        self.init_result = init_result
        self.dva = dva
        self.zero_dva = zero_dva
        self.config = None
        self.raw = None

    def initialize(self, config):
        self.config = config
        return self.init_result

    def empty(self, shape, *, dtype, pin_memory):
        assert dtype is torch.int8
        assert pin_memory is True
        self.raw = torch.empty(shape, dtype=dtype)
        return self.raw

    def get_dva(self, address):
        if self.zero_dva:
            return 0
        return self.dva + (address - self.raw.data_ptr())


def _install_fake_ddr(monkeypatch, fake):
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", types.SimpleNamespace(offload=fake))
    monkeypatch.setitem(sys.modules, "torch_npu", types.SimpleNamespace(npu=types.SimpleNamespace(current_device=lambda: 3)))


def test_ddr_pool_uses_aligned_registration_and_dva(monkeypatch):
    fake = _FakeOffload(dva=0x50000000)
    _install_fake_ddr(monkeypatch, fake)

    pools = make_source_pools(3, 7, 5, ddr=True)

    assert fake.config.device_id == 3
    assert fake.config.world_size == 1
    assert fake.config.rank_id == 0
    assert fake.config.scene == "local"
    assert fake.config.flags == fake.OFFLOAD_FLAG_GIANT_PAGE
    assert fake.config.reserve_size == 3 * 7 + 3 * 5 + 3 * 2 * 1024 * 1024
    assert fake.config.reserve_size == fake.config.alloc_size
    assert pools.k.data_ptr() % (2 * 1024 * 1024) == 0
    assert pools.v.data_ptr() % (2 * 1024 * 1024) == 0
    assert pools.k_address == fake.dva + (pools.k.data_ptr() - fake.raw.data_ptr())
    assert pools.k_address != pools.k.data_ptr()
    assert pools.v_address - pools.k_address >= 2 * 1024 * 1024
    assert torch.all((pools.k >= 0) & (pools.k <= 255))
    fake.raw = None
    assert int(pools.k[0, 0]) >= 0


def test_ddr_missing_get_dva_is_rejected(monkeypatch):
    fake = _FakeOffload()
    fake.get_dva = None
    _install_fake_ddr(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="get_dva"):
        make_source_pools(1, 8, 4, ddr=True)


def test_ddr_missing_memfabric_is_rejected(monkeypatch):
    _install_fake_ddr(monkeypatch, _FakeOffload())
    monkeypatch.setitem(sys.modules, "memfabric_hybrid", None)
    with pytest.raises(RuntimeError, match="requires"):
        make_source_pools(1, 8, 4, ddr=True)


def test_ddr_initialization_failure_is_not_silently_fallback(monkeypatch):
    fake = _FakeOffload(init_result=7)
    _install_fake_ddr(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="initialize"):
        make_source_pools(1, 8, 4, ddr=True)


def test_ddr_zero_dva_is_rejected(monkeypatch):
    fake = _FakeOffload(zero_dva=True)
    _install_fake_ddr(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="DVA"):
        make_source_pools(1, 8, 4, ddr=True)


@pytest.mark.parametrize("token_count,k_bytes,v_bytes", [(0, 8, 4), (1, 0, 4), (1, 8, -1)])
def test_pool_sizes_must_be_positive(token_count, k_bytes, v_bytes):
    with pytest.raises(ValueError):
        make_source_pools(token_count, k_bytes, v_bytes, ddr=False, device="cpu")
