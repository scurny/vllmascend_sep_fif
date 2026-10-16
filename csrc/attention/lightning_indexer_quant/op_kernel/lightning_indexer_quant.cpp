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
#include "lib/matmul_intf.h"
#include "arch35/lightning_indexer_quant_kernel_arch35.h"
#include "lightning_indexer_quant_template_tiling_key.h"

using namespace LIQKernel;

#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                                                         \
    do {                                                                                                     \
        templateClass<LIQType<__VA_ARGS__>> op;                                                              \
        GET_TILING_DATA_WITH_STRUCT(LIQTilingData, tiling_data_in, tiling);                                  \
        const LIQTilingData *__restrict tiling_data = &tiling_data_in;                                       \
        /* LIQ 调试标记: 把 kernel 读到的 emaThreshold 位型写到 out[0].                                     \
           NOTE: 在 op.Init() 之前禁止 return! MIX_AIC_1_2 下 AIC 主核提前退出会导致             \
           AIV 从核等待协议信号挂死 (aicore 异常). 合法的提前退出位置是 Process() 开头. */          \
        union {                                                                                              \
            float f;                                                                                         \
            int32_t i;                                                                                       \
        } liqDbgEma;                                                                                         \
        liqDbgEma.f = tiling_data->emaThreshold;                                                             \
        *(reinterpret_cast<__gm__ int32_t *>(sparseIndices)) = liqDbgEma.i;                                  \
        op.Init(query, key, weights, queryScale, keyScale, actualSeqLengthsQ, actualSeqLengthsK, blocktable, \
                sparseIndices, user, tiling_data, &tPipe,                                                    \
                tokenGvaTable, prefetchStagingBuf, prefetchSlotCounter, tokenToPrefetchSlot);                \
        op.Process();                                                                                        \
    } while (0)

template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int Q_LAYOUT_T, int K_LAYOUT_T>
// NOTE: 实参顺序必须与 op_host/lightning_indexer_quant_def.cpp 的声明一致
// (inputs 按定义序 + outputs + workspace + tiling), 否则运行时传参错位,
// 曾导致 tiling 槽位收到 NULL/prefetch_slot_counter 指针而 aicore 95/271 崩溃.
__global__ __aicore__ void lightning_indexer_quant(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
                                                   __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale,
                                                   __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengthsK,
                                                   __gm__ uint8_t *blocktable, __gm__ uint8_t *tokenGvaTable,
                                                   __gm__ uint8_t *prefetchStagingBuf, __gm__ uint8_t *prefetchSlotCounter,
                                                   __gm__ uint8_t *tokenToPrefetchSlot, __gm__ uint8_t *sparseIndices,
                                                   __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
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
