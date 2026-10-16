from dataclasses import dataclass

import torch


_ALIGNMENT = 2 * 1024 * 1024


@dataclass(frozen=True)
class SourcePools:
    k: torch.Tensor
    v: torch.Tensor
    k_address: int
    v_address: int

    def gva(self, physical_tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(physical_tokens, torch.Tensor) or physical_tokens.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise TypeError("physical_tokens must be an integer tensor")
        tokens = physical_tokens.to(dtype=torch.int64, device="cpu")
        if torch.any(tokens < 0) or torch.any(tokens >= self.k.shape[0]):
            raise IndexError("physical_tokens contains an out-of-range token")
        result = torch.empty((*tokens.shape, 2), dtype=torch.int64)
        result[..., 0] = self.k_address + tokens * self.k.shape[-1]
        result[..., 1] = self.v_address + tokens * self.v.shape[-1]
        return result


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _hbm_pool(token_count: int, k_bytes: int, v_bytes: int, device: str) -> SourcePools:
    k = torch.randint(0, 256, (token_count, k_bytes), dtype=torch.uint8, device=device)
    v = torch.randint(0, 256, (token_count, v_bytes), dtype=torch.uint8, device=device)
    return SourcePools(k, v, int(k.data_ptr()), int(v.data_ptr()))


def _ddr_pool(token_count: int, k_bytes: int, v_bytes: int) -> SourcePools:
    try:
        import torch_npu
        from memfabric_hybrid import offload
    except ImportError as exc:
        raise RuntimeError("DDR source mode requires torch_npu and memfabric_hybrid.offload") from exc

    giant_page = getattr(offload, "OFFLOAD_FLAG_GIANT_PAGE", None)
    scene = getattr(getattr(offload, "Scene", None), "LOCAL", None)
    config_type = getattr(offload, "OffloadConfig", None)
    if giant_page is None or scene is None or config_type is None or not callable(getattr(offload, "get_dva", None)):
        raise RuntimeError("memfabric_hybrid.offload lacks rank-local DDR registration APIs, including get_dva")

    k_total = token_count * k_bytes
    v_total = token_count * v_bytes
    k_chunks = (k_total + _ALIGNMENT - 1) // _ALIGNMENT
    v_chunks = (v_total + _ALIGNMENT - 1) // _ALIGNMENT
    backing_bytes = (1 + k_chunks + v_chunks) * _ALIGNMENT
    reserve_bytes = k_total + v_total + 3 * _ALIGNMENT
    config = config_type()
    config.device_id = torch_npu.npu.current_device()
    config.reserve_size = reserve_bytes
    config.alloc_size = reserve_bytes
    config.world_size = 1
    config.rank_id = 0
    config.scene = scene
    config.flags = giant_page
    result = offload.initialize(config)
    if result != 0:
        raise RuntimeError(f"failed to initialize DDR offload pool (status={result})")

    raw = offload.empty([backing_bytes], dtype=torch.int8, pin_memory=True)
    base = (int(raw.data_ptr()) + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
    offset = base - int(raw.data_ptr())
    k_raw = raw[offset : offset + k_total]
    v_start = offset + k_chunks * _ALIGNMENT
    v_raw = raw[v_start : v_start + v_total]
    k = k_raw.view(torch.uint8).reshape(token_count, k_bytes)
    v = v_raw.view(torch.uint8).reshape(token_count, v_bytes)
    k.copy_(torch.randint(0, 256, k.shape, dtype=torch.uint8))
    v.copy_(torch.randint(0, 256, v.shape, dtype=torch.uint8))
    k_address = int(offload.get_dva(int(k.data_ptr())))
    v_address = int(offload.get_dva(int(v.data_ptr())))
    if k_address == 0 or v_address == 0:
        raise RuntimeError("memfabric_hybrid returned a zero DVA for DDR source pool")
    return SourcePools(k, v, k_address, v_address)


def make_source_pools(
    token_count: int,
    k_bytes: int,
    v_bytes: int,
    *,
    ddr: bool,
    device: str = "npu",
) -> SourcePools:
    _require_positive("token_count", token_count)
    _require_positive("k_bytes", k_bytes)
    _require_positive("v_bytes", v_bytes)
    if ddr:
        return _ddr_pool(token_count, k_bytes, v_bytes)
    return _hbm_pool(token_count, k_bytes, v_bytes, device)
