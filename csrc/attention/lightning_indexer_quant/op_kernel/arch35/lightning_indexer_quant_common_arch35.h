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
 * \file lightning_indexer_quant_common.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_QUANT_COMMON_H
#define LIGHTNING_INDEXER_QUANT_COMMON_H

namespace LIQCommon {

// 与tiling的layout保持一致
enum class LI_LAYOUT : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BSND = 2
};

// ---- 崩溃二分用的阶段开关: 测试侧把 ema_threshold 传成下列负值, kernel 在对应阶段提前退出。
// 仅用于 950 bring-up 调试, 定位完成后移除。 ----
constexpr float LIQ_DEBUG_STAGE_ENTRY_ONLY = -1.0f;  // 仅入口 + tiling 解析
constexpr float LIQ_DEBUG_STAGE_AFTER_INIT = -2.0f;  // + Init (InitBuffers/SplitCore/标量GM读)
constexpr float LIQ_DEBUG_STAGE_AFTER_MAIN = -3.0f;  // + ProcessMain (AIC matmul / AIV Vec0/Vec1)
// Init 内部里程碑 (在 Init 内 return, 由 Process 顶部门完成合法退出):
constexpr float LIQ_DEBUG_INIT_TILING = -11.0f;      // InitTilingData + InitActualSeqLen 后
constexpr float LIQ_DEBUG_INIT_SPLIT = -12.0f;       // + SplitCore 后
constexpr float LIQ_DEBUG_INIT_BUFBOUND = -13.0f;    // + 全部 SetGlobalBuffer/InitParams 后 (InitBuffers 前)
constexpr float LIQ_DEBUG_INIT_FULL = -14.0f;        // + InitBuffers (完整 Init, 等价 -1/-2)
constexpr float LIQ_DEBUG_MAIN_FULL = -15.0f;        // + ProcessMain (等价 -3)
// 数值调试: 跑完整流水, 但 Vec1 在读入 mm1Res 后把原始 2048 float 位型直接写入输出张量,
// host 侧按 128-token tile 解码比对, 定位 cube 落盘丢失 (不执行排序/输出)
constexpr float LIQ_DEBUG_DUMP_MM1 = -20.0f;

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
          LI_LAYOUT Q_LAYOUT_T = LI_LAYOUT::BSND, LI_LAYOUT K_LAYOUT_T = LI_LAYOUT::PA_BSND, typename... Args>
struct LIQType {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr LI_LAYOUT layout = Q_LAYOUT_T;
    static constexpr LI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t tensorWeightsOffset;
    uint64_t indiceOutOffset;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
    bool isValid = false;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // 950融合核UB直通路径的核间同步 (AIC fixpipe直写配对AIV的UB, ping-pong按chunk轮转)
    static constexpr uint32_t FIA_SYNC_MODE4 = 4;
    // AIV->AIC "slot已消费" 事件基址: id = BASE + slot*2 + aiv (aiv = blockIdx%2)
    static constexpr uint32_t SYNC_VC_EVENT_BASE = 4U;
    // AIC->AIV "slot已生产" 事件基址: id = BASE + slot*2 + aiv
    static constexpr uint32_t SYNC_CV_EVENT_BASE = 8U;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncC1V0 = 2U;
    uint32_t syncV1C1 = 0U;
    uint32_t syncV0C1 = 1U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;              // topK选取大小
    uint64_t kSeqSize = 0ULL;          // kv最大S长度
    uint64_t qSeqSize = 1ULL;          // q最大S长度
    uint32_t kCacheBlockSize = 0;      // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0;  // PA场景的最大单batch block number
    LI_LAYOUT outputLayout;            // 输出的格式
    bool attenMaskFlag = false;

    uint32_t actualLenQDims = 0U;  // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;   // KV 的actualSeqLength 的维度
    bool isAccumSeqS1 = false;     // 是否累加模式
    bool isAccumSeqS2 = false;     // 是否累加模式

    // ---- ECHO-style prefetch ----
    float emaThreshold = 0.0f;         // EMA 预测阈值，≤0 时不触发 prefetch
    uint32_t prefetchCandidateMax = 0; // 最大候选 token 数
    uint32_t kEntryBytes = 0;          // 单 token KV 字节数 (K+V)
    uint32_t tokenSizeBytesK = 0;      // 单 token K 字节数 (用于查 token_gva_table K/V 分拆)
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U;  // S2的起始位置
    uint32_t s2End = 0U;    // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;  // 当前核是否需要进行Decode归约任务
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}
}  // namespace LIQCommon

#endif  // LIGHTNING_INDEXER_QUANT_COMMON_H