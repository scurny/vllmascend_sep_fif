/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_quant.cpp
 * \brief
 */

#include "kernel_operator.h"
#if defined(LIQ_PREFETCH_DIAGNOSTICS) && LIQ_PREFETCH_DIAGNOSTICS
#include "utils/debug/asc_printf.h"
#endif
#include "lib/matmul_intf.h"
#include "arch35/lightning_indexer_quant_kernel_arch35.h"
#include "lightning_indexer_quant_template_tiling_key.h"

using namespace LIQKernel;

#if defined(LIQ_PREFETCH_DIAGNOSTICS) && LIQ_PREFETCH_DIAGNOSTICS
#define LIQ_PREFETCH_ENTRY_DIAGNOSTICS(tilingData, gvaTable, stagingBuf, slotCounter, slotMap)              \
    do {                                                                                                      \
        if (ASCEND_IS_AIV && GetBlockIdx() == 0) {                                                           \
            AscendC::printf("[LIQ_DIAG] entry v1 built=%s %s ema=%f max=%u entry_k=%u gva=%d staging=%d counter=%d " \
                            "slot=%d\n",                                                                      \
                   __DATE__, __TIME__, static_cast<float>((tilingData)->emaThreshold),                      \
                   static_cast<uint32_t>((tilingData)->prefetchCandidateMax),                                \
                   static_cast<uint32_t>((tilingData)->tokenSizeBytesK),                                    \
                   static_cast<int32_t>((gvaTable) != nullptr), static_cast<int32_t>((stagingBuf) != nullptr), \
                   static_cast<int32_t>((slotCounter) != nullptr), static_cast<int32_t>((slotMap) != nullptr)); \
        }                                                                                                     \
    } while (0)
#else
#define LIQ_PREFETCH_ENTRY_DIAGNOSTICS(tilingData, gvaTable, stagingBuf, slotCounter, slotMap) do { } while (0)
#endif

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<LIQType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(LIQTilingData, tiling_data_in, tiling);                                  \
        const LIQTilingData *__restrict tiling_data = &tiling_data_in;                                       \
        LIQ_PREFETCH_ENTRY_DIAGNOSTICS(tiling_data, tokenGvaTable, prefetchStagingBuf,                      \
                                       prefetchSlotCounter, tokenToPrefetchSlot);                            \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                sparseIndices, user, tiling_data, &tPipe,                                                    \
                tokenGvaTable, prefetchStagingBuf, prefetchSlotCounter, tokenToPrefetchSlot);                \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
__global__ __aicore__ void lightning_indexer_quant(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                   __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                   __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *tokenGvaTable,
                                                   __gm__ uint8_t *prefetchStagingBuf,
                                                   __gm__ uint8_t *prefetchSlotCounter,
                                                   __gm__ uint8_t *tokenToPrefetchSlot,
                                                   __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                                   __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 200)
    // A2 不编
#else
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    INVOKE_LI_NO_KFC_OP_IMPL(LIQPreload, int8_t, int8_t, int32_t,
                             PAGE_ATTENTION, LI_LAYOUT(Q_LAYOUT_T), LI_LAYOUT(K_LAYOUT_T));
#endif
}
