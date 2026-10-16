/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License in the root of the software repository for details.
 */

#ifndef LIGHTNING_INDEXER_QUANT_TILING_LIMITS_H_
#define LIGHTNING_INDEXER_QUANT_TILING_LIMITS_H_

#include <cstdint>

namespace optiling {
constexpr uint32_t LIQ_QUERY_HEAD_COUNT = 64;
// Matches arch35 LIQVector::InitBuffers at 64 query heads.
constexpr uint64_t LIQ_BASE_UB_BYTES =
    16ULL * 8ULL + 2ULL * 2048ULL * 4ULL * 2ULL + 2048ULL * 4ULL + 2048ULL * 4ULL + 64ULL * 1024ULL +
    2ULL * 4096ULL * 4ULL;
// Extra LIQ margin, independent of reservations already deducted by the platform.
constexpr uint64_t LIQ_UB_SAFETY_RESERVE_BYTES = 8192;
constexpr uint64_t LIQ_PREFETCH_MASK_BYTES = 256;
constexpr uint64_t LIQ_REQUIRED_UB_BYTES = LIQ_BASE_UB_BYTES + LIQ_UB_SAFETY_RESERVE_BYTES;
constexpr uint64_t LIQ_PREFETCH_REQUIRED_UB_BYTES = LIQ_REQUIRED_UB_BYTES + LIQ_PREFETCH_MASK_BYTES;

constexpr bool IsSupportedLiqQueryHeadCount(int64_t query_head_count)
{
    return query_head_count == static_cast<int64_t>(LIQ_QUERY_HEAD_COUNT);
}

constexpr bool HasLiqBaseUbCapacity(uint64_t ub_size)
{
    return ub_size >= LIQ_REQUIRED_UB_BYTES;
}

constexpr bool HasLiqPrefetchUbCapacity(uint64_t ub_size, uint64_t entry_bytes)
{
    return ub_size >= LIQ_PREFETCH_REQUIRED_UB_BYTES &&
           entry_bytes <= ub_size - LIQ_PREFETCH_REQUIRED_UB_BYTES;
}
}

#endif
