/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file lightning_indexer_quant_service_vector.h
 * \brief 950(arch35) vector: Vec1从自身UB消费AIC fixpipe直写的Qk(int32)结果,
 *        完成加权归约(w'·relu(fp16(qk·2^-10))累加)后接原排序/归并/LD管线。
 */
#ifndef LIGHTNING_INDEXER_QUANT_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_QUANT_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_quant_common_arch35.h"
#include "lightning_indexer_quant_vector_arch35.h"

namespace LIQKernel {
using namespace LIQCommon;
using namespace LIQServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;

template <typename LIQT>
class LIQVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = LIQT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIQT::keyLayout;
    static constexpr bool PAGE_ATTENTION = LIQT::pageAttention;

    __aicore__ inline LIQVector(){};
    __aicore__ inline void ProcessVec1(const LIQCommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LIQCommon::ConstInfo &constInfo,
                                      const LIQTilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<float> vec1ResGm, GlobalTensor<int64_t> vec1ParamGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    // ---- ECHO-style prefetch ----
    __aicore__ inline void InitPrefetchTensors(GlobalTensor<uint64_t> tokenGvaTableGm,
                                                GlobalTensor<uint8_t> prefetchStagingBufGm,
                                                GlobalTensor<int32_t> prefetchSlotCounterGm,
                                                GlobalTensor<int32_t> tokenToPrefetchSlotGm);
    __aicore__ inline void PrefetchScoresInVec1(const LocalTensor<float> &scoresUb,
                                                const LIQCommon::RunInfo &info,
                                                uint32_t cuS2Len, uint32_t cuBaseS2Idx);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // ---- ECHO-style prefetch GM tensors ----
    GlobalTensor<uint64_t> tokenGvaTableGm_;
    GlobalTensor<uint8_t> prefetchStagingBufGm_;
    GlobalTensor<int32_t> prefetchSlotCounterGm_;
    GlobalTensor<int32_t> tokenToPrefetchSlotGm_;
    // =================================常量区=================================
    // ---- 950 UB直通: 与cube侧 S1G/S2_BASIC_BLOCK_L0 及 QK slot布局严格一致 ----
    static constexpr uint32_t S1G_L0_BLOCK = 128;
    static constexpr uint32_t S2_L0_BLOCK = 128;
    static constexpr uint64_t QK_SLOT_ELEMS = (S1G_L0_BLOCK / 2) * S2_L0_BLOCK;  // 64行*128列 int32
    static constexpr uint64_t QK_UB_BYTES = 2 * QK_SLOT_ELEMS * sizeof(int32_t);
    static constexpr float QK_DEQ_SCALE = 1.0f / 1024.0f;  // 旧DEQF16的2^-10反量化

private:
    __aicore__ inline void GetKeyScale(const LIQCommon::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    __aicore__ inline void ConsumeQkChunks(const LIQCommon::RunInfo &info, int32_t aivIdx, int32_t cuBaseS1Idx);
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // 950 UB直通: 必须是AIV侧TPipe的第一个VECCALC缓冲, 与AIC侧bufQkUb_地址一致
    TBuf<TPosition::VECCALC> qkBuf_;
    LocalTensor<int32_t> qkUb_;

    // Qk归约工作区: acc[本核token数][s2Base] + rowF32[128] + rowF16[128]
    TBuf<TPosition::VECCALC> qkWorkBuf_;
    LocalTensor<float> qkAcc_;
    LocalTensor<float> qkRowF32_;
    LocalTensor<half> qkRowF16_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    // ---- ECHO-style prefetch ----
    uint32_t prefetchCandidateMax_ = 0;
    uint32_t kEntryBytes_ = 0;
    uint32_t tokenSizeBytesK_ = 0;
    TBuf<TPosition::VECCALC> prefetchUbBuf_;
    LocalTensor<uint8_t> prefetchUbLocal_;
    TBuf<TPosition::VECCALC> cmpMaskBuf_;
    LocalTensor<uint8_t> cmpMaskUb_;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;

    struct LIQCommon::ConstInfo constInfo_;
};

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::GetKeyScale(const LIQCommon::RunInfo &runInfo, const LocalTensor<half> &resUb,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb, kScaleGm[blockId * kCacheBlockSize_ + startBlockTableOffset],
                                 copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_], kScaleGm[blockId * kCacheBlockSize_],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::InitBuffers(TPipe *pipe)
{
    // 950 UB直通: 第一个VECCALC缓冲, 与AIC侧bufQkUb_基地址一致 (融合核fixpipe跨核落址)
    pipe->InitBuffer(qkBuf_, QK_UB_BYTES);
    qkUb_ = qkBuf_.Get<int32_t>();

    // Qk归约工作区: acc[本核token数=s1Base/2][s2Base] fp32 + 行缓冲
    pipe->InitBuffer(qkWorkBuf_, CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(float) + S2_L0_BLOCK * sizeof(float) +
                                        S2_L0_BLOCK * sizeof(half));
    LocalTensor<float> workF32 = qkWorkBuf_.Get<float>();
    qkAcc_ = workF32;                                                       // [s1Base/2][s2Base]
    qkRowF32_ = workF32[CeilDiv(s1BaseSize_, 2) * s2BaseSize_];             // [128]
    qkRowF16_ = workF32.ReinterpretCast<half>()[(CeilDiv(s1BaseSize_, 2) * s2BaseSize_ + S2_L0_BLOCK) * 2];

    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));                                        // 1 KB
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2);                                     // 32KB
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));                                          // 8 KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                         // 8 KB
    pipe->InitBuffer(tmpBuf_, 64 * 1024);                                                               // 64KB
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE * sizeof(float));  // 32KB

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);

    // step3. 初始化vec1ParamGm，是否进行LD的标志位设为-1(needFd=-1)
    // vec1ResIn32Gm = [aic, 2, s1BaseSize_, 16] int32
    // ws清零 [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, ......]
    LocalTensor<float> tmpfBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpfBuff.template ReinterpretCast<int32_t>(), -1, 2 * paramNum_);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    // 交错token分配: 本核负责绝对S1行 tLocal = aivIdx, aivIdx+2, ...
    int32_t aivIdx = blockId_ % 2;
    for (int32_t t = aivIdx; t < (int32_t)s1BaseSize_; t += 2) {
        int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ + t * 2 * paramNum_;
        // 256B整块, 用裸DataCopy规避DataCopyPad在950上的UB→GM落址偏移问题
        DataCopy(vec1ParamGm[wsInfoOffset], tmpfBuff.template ReinterpretCast<int64_t>(), 2 * paramNum_);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(tmpfBuff);
    // ---- ECHO-style prefetch UB buffer ----
    if (prefetchCandidateMax_ > 0 && kEntryBytes_ > 0) {
        pipe->InitBuffer(prefetchUbBuf_, kEntryBytes_);
        prefetchUbLocal_ = prefetchUbBuf_.Get<uint8_t>();
        pipe->InitBuffer(cmpMaskBuf_, 32);
        cmpMaskUb_ = cmpMaskBuf_.Get<uint8_t>();
    }
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, BASE_TOPK_VALUE_IDX_SIZE * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, BASE_TOPK_VALUE_IDX_SIZE * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::InitParams(const struct LIQCommon::ConstInfo &constInfo,
                                                   const LIQTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 2048
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    // ---- ECHO-style prefetch ----
    prefetchCandidateMax_ = constInfo.prefetchCandidateMax;
    kEntryBytes_ = constInfo.kEntryBytes;
    tokenSizeBytesK_ = constInfo.tokenSizeBytesK;
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                                           GlobalTensor<half> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::InitVecWorkspaceTensor(GlobalTensor<float> vec1ResGm,
                                                               GlobalTensor<int64_t> vec1ParamGm)
{
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::InitPrefetchTensors(
    GlobalTensor<uint64_t> tokenGvaTableGm, GlobalTensor<uint8_t> prefetchStagingBufGm,
    GlobalTensor<int32_t> prefetchSlotCounterGm, GlobalTensor<int32_t> tokenToPrefetchSlotGm)
{
    this->tokenGvaTableGm_ = tokenGvaTableGm;
    this->prefetchStagingBufGm_ = prefetchStagingBufGm;
    this->prefetchSlotCounterGm_ = prefetchSlotCounterGm;
    this->tokenToPrefetchSlotGm_ = tokenToPrefetchSlotGm;
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::PrefetchScoresInVec1(
    const LocalTensor<float> &scoresUb, const LIQCommon::RunInfo &info,
    uint32_t cuS2Len, uint32_t cuBaseS2Idx)
{
    // 仅 emaThreshold > 0 且 prefetch 输入已提供时触发 prefetch
    if (constInfo_.emaThreshold <= 0.0f || prefetchCandidateMax_ == 0 || kEntryBytes_ == 0) {
        return;
    }

    if (cuS2Len == 0) {
        return;
    }

    //---- Per-core slot range: 每个 AIV 核心拥有独占的 slot 区间 ----
    int32_t numCores = AscendC::GetBlockNum () * 2;  // AIV 核数 = AIC * 2
    int32_t slotsPerCore = prefetchCandidateMax_ / numCores;

    if (slotsPerCore <= 0) {
        return;
    }

    int32_t coreSlotBase = blockId_ * slotsPerCore;
    int32_t coreSlotEnd = coreSlotBase + slotsPerCore;
    int32_t localSlot = coreSlotBase;

    // CompareScalar: scoresUb > emaThreshold → bitmask
    uint32_t cmpLenAlign = LIQCommon::Align(cuS2Len, (uint32_t)32);
    AscendC::CompareScalar(cmpMaskUb_, scoresUb, static_cast<float>(constInfo_.emaThreshold),
    AscendC::CMPMODE::GT, cmpLenAlign);

    // 收集超过阈值的 position
    uint32_t validInRow = 0;
    uint32_t bitPositions [256];  //s2BaseSize 上限 2048, 但单次 cuS2Len <= 2048
    for (uint32_t j = 0; j < cuS2Len; j++) {
        if ((cmpMaskUb_.GetValue(j / 8) >> (j % 8)) & 1) {
            bitPositions[validInRow++] = j;
        }
    }

    if (validInRow == 0) {
        return;
    }

    // 检查核内 slot capacity
    int32_t slotStart = localSlot;
    localSlot += static_cast<int32_t>(validInRow);

    if (slotStart + static_cast<int32_t>(validInRow) > coreSlotEnd) {
        return;  // 放弃这行
    }

    //---- 跨 request 偏移: tokenGvaTable/tokenToPrefetchSlot layout 为 [num_reqs, max_tokens, *] ----
    int64_t reqTokenBase = static_cast<int64_t>(info.bIdx) * static_cast<int64_t>(constInfo_.kSeqSize);
    uint32_t tokenSizeBytesV = kEntryBytes_ - tokenSizeBytesK_;
    AscendC::DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyParamsK(1, tokenSizeBytesK_, 0, 0, 0);
    AscendC::DataCopyExtParams copyParamsV(1, tokenSizeBytesV, 0, 0, 0);
    AscendC::DataCopyExtParams copyParamsFull(1, kEntryBytes_, 0, 0, 0);

    // 逐 token: 两次 GVA 查表 + 两次 DataCopyPad (K/V 分拆，MTE 流水线)
    for (uint32_t k = 0; k < validInRow; k++) {
        int64_t kvPos = cuBaseS2Idx + static_cast<int64_t>(bitPositions[k]);
        int64_t globalPos = reqTokenBase + kvPos;
        int32_t slot = slotStart + static_cast<int32_t>(k);

        // 1. 从 tokenGvaTable 读取 K/V 两个 GVA 地址 (layout: [num_reqs, max_tokens, 2])
        uint64_t gvaK = tokenGvaTableGm_.GetValue(static_cast<uint64_t>(globalPos) * 2);
        uint64_t gvaV = tokenGvaTableGm_.GetValue(static_cast<uint64_t>(globalPos) * 2 + 1);

        // 2. 写 token→slot 映射
        tokenToPrefetchSlotGm_.SetValue(static_cast<uint64_t>(globalPos), slot);

        // 3. GVA→UB: 先 K 再 V
        GlobalTensor<uint8_t> srcGmK;
        srcGmK.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gvaK));
        AscendC::DataCopyPad(prefetchUbLocal_, srcGmK, copyParamsK, padParams);
        GlobalTensor<uint8_t> srcGmV;
        srcGmV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(gvaV));
        AscendC::DataCopyPad(prefetchUbLocal_[tokenSizeBytesK_], srcGmV, copyParamsV, padParams);
        // MTE3(UB→GM) 必须等 MTE2(GM→UB) 装填完成, 否则搬到 staging 的是上一个 token 的残影
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);

        // 4. UB→Staging Buffer
        AscendC::DataCopyPad(
        prefetchStagingBufGm_[static_cast<uint64_t>(slot) * static_cast<uint64_t>(kEntryBytes_)],
        prefetchUbLocal_, copyParamsFull);
        // 下一轮 MTE2 重写 UB 前必须等 MTE3 读走, 防止覆盖未搬完的数据
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    }
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::AllocEventID()
{
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::FreeEventID()
{
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LIQServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
}

// 950 UB直通: 消费本s2块全部Qk块(与AIC侧CalcMmInfo严格同序), 对本核token做加权归约:
// acc[s2] = Σ_g w'[g] · relu(fp16(qk[g][s2] · 2^-10)),  w' = fp16(w·qScale)
// (数值语义与旧arch22链路 DEQF16+reluPre+Ws-Mmad 逐点一致)
// 注意: 本函数的token归属(tLocal交错分配)与ProcessVec1一致, 要求tokensPerHalf==1, 即gSize==64
template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::ConsumeQkChunks(const LIQCommon::RunInfo &info, int32_t aivIdx,
                                                        int32_t cuBaseS1Idx)
{
    uint32_t tokensInBlock = info.actMBaseSize / gSize_;
    uint32_t s2L0LoopCnt = CeilDiv(info.actualSingleProcessSInnerSize, S2_L0_BLOCK);
    uint32_t s1L0LoopCnt = CeilDiv(info.actMBaseSize, S1G_L0_BLOCK);
    uint32_t totalChunks = s2L0LoopCnt * s1L0LoopCnt;
    // 每个AIV半个L0块(64行)包含的token数; token边界与dualDstCtl的M/2拆分对齐;
    // gSize!=64时token归属与ProcessVec1交错分配不一致, 不在本设计支持范围
    uint32_t tokensPerHalf = (S1G_L0_BLOCK / 2) / gSize_;
    if (tokensPerHalf == 0) {
        tokensPerHalf = 1;  // gSize>64的形状不在本设计支持范围
    }

    // 加载w' = fp16(w·qScale), 经inQueue_做MTE2→V同步
    LocalTensor<half> wInUb = inQueue_.AllocTensor<half>();
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = gSize_ * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    int32_t myTokenNum = 0;
    for (uint32_t t = aivIdx; t < tokensInBlock; t += 2) {
        int64_t wGmOffset = info.tensorWeightsOffset + (cuBaseS1Idx + t) * qHeadNum_;
        AscendC::DataCopyPad(wInUb[myTokenNum * 2 * gSize_], weightsGm[wGmOffset], copyInParams, padParams);
        AscendC::DataCopyPad(wInUb[myTokenNum * 2 * gSize_ + gSize_], qScaleGm[wGmOffset], copyInParams, padParams);
        myTokenNum++;
    }
    inQueue_.EnQue<half>(wInUb);
    wInUb = inQueue_.DeQue<half>();
    for (int32_t ti = 0; ti < myTokenNum; ti++) {
        AscendC::Mul(wInUb[ti * 2 * gSize_], wInUb[ti * 2 * gSize_], wInUb[ti * 2 * gSize_ + gSize_], gSize_);
    }
    PipeBarrier<PIPE_V>();  // 确保Mul结果对chunk循环内的标量GetValue可见; wInUb保持占用至chunk循环结束

    // acc清零 (尾部未覆盖列保持0, 排序阶段按cuS2Len掩蔽)
    for (int32_t ti = 0; ti < myTokenNum; ti++) {
        AscendC::Duplicate(qkAcc_[ti * s2BaseSize_], 0.0f, s2BaseSize_);
    }
    PipeBarrier<PIPE_V>();

    for (uint32_t chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
        uint32_t slot = chunkIdx % 2;
        uint32_t s2L0LoopId = chunkIdx / s1L0LoopCnt;
        uint32_t s1gL0LoopId = chunkIdx % s1L0LoopCnt;
        CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_V>(
            LIQCommon::ConstInfo::SYNC_CV_EVENT_BASE + slot * 2 + aivIdx);
        uint32_t colBase = s2L0LoopId * S2_L0_BLOCK;
        uint32_t nReal = info.actualSingleProcessSInnerSize > colBase
                             ? LIQCommon::Min(S2_L0_BLOCK, info.actualSingleProcessSInnerSize - colBase)
                             : 0;
        if (nReal > 0) {
            // dualDstCtl=1: 本核UB slot内即为归属本核的64行(其行0对应全局M行aivIdx*64), 无需再加行偏移
            LocalTensor<int32_t> qk = qkUb_[slot * QK_SLOT_ELEMS];
            // 本chunk归属本核的token: tLocal = s1gL0LoopId*(S1G_L0/gSize) + aivIdx*tokensPerHalf + j
            for (uint32_t j = 0; j < tokensPerHalf; j++) {
                uint32_t tLocal = s1gL0LoopId * (S1G_L0_BLOCK / gSize_) + aivIdx * tokensPerHalf + j;
                if (tLocal >= tokensInBlock) {
                    break;
                }
                int32_t ti = static_cast<int32_t>(s1gL0LoopId * tokensPerHalf + j);  // 本核acc行号(稳定)
                LocalTensor<float> acc = qkAcc_[ti * s2BaseSize_ + colBase];
                for (int32_t g = 0; g < gSize_; g++) {
                    float wVal = static_cast<float>(wInUb.GetValue(ti * 2 * gSize_ + g));
                    // S = relu(fp16(qk·2^-10)), 累加 w'[g]·S (fp32)
                    AscendC::Cast(qkRowF32_, qk[(j * gSize_ + g) * S2_L0_BLOCK], RoundMode::CAST_NONE, nReal);
                    AscendC::Muls(qkRowF32_, qkRowF32_, QK_DEQ_SCALE, nReal);
                    AscendC::Cast(qkRowF16_, qkRowF32_, RoundMode::CAST_RINT, nReal);
                    AscendC::Cast(qkRowF32_, qkRowF16_, RoundMode::CAST_NONE, nReal);
                    AscendC::Maxs(qkRowF32_, qkRowF32_, 0.0f, nReal);
                    AscendC::Muls(qkRowF32_, qkRowF32_, wVal, nReal);
                    AscendC::Add(acc, acc, qkRowF32_, nReal);
                }
            }
        }
        CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_V>(
            LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + slot * 2 + aivIdx);
    }
    inQueue_.FreeTensor(wInUb);
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::ProcessVec1(const LIQCommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;
    int32_t aivIdx = blockId_ % 2;

    // ---- 950 UB直通: 消费本s2块的Qk结果并完成加权归约 ----
    ConsumeQkChunks(info, aivIdx, cuBaseS1Idx);

    // 交错token分配(与fixpipe dualDstCtl的M/2拆分一致): 本核token tLocal = aivIdx, aivIdx+2, ...
    int32_t cuS1ProcNum =
        cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t myTokenNum = cuS1ProcNum > aivIdx ? (cuS1ProcNum - aivIdx + 1) / 2 : 0;

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK_VALUE_IDX_SIZE);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }

    for (int32_t ti = 0; ti < myTokenNum; ti++) {
        int32_t tLocal = aivIdx + 2 * ti;
        int32_t cuS1Idx = cuBaseS1Idx + tLocal;
        // cuRealAcSeq: 当前token对应的AcSeq
        int32_t cuRealAcSeq = info.actS2Size;
        if (constInfo_.attenMaskFlag) {
            // attenMask true场景
            cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1Idx);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = mmInUb[cuS2LenVecAlign];
            LocalTensor<half> kScaleTUb = kScaleUb.template ReinterpretCast<half>()[cuS2LenVecAlign];
            AscendC::DataCopyPadExtParams<half> padTParams{false, 0, 0, 0};
            GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, cuS2Len);
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            // UB直通: 本token的C_ws已在qkAcc_[ti]完成
            LocalTensor<float> scoreUb = qkAcc_[ti * s2BaseSize_];
            if (constInfo_.emaThreshold == LIQCommon::LIQ_DEBUG_DUMP_MM1) {
                // 调试(-20): 把本核归约出的C_ws原始位型 (2048 float, 乘kScale之前) 直接写入
                // 输出张量, host 侧按 128-token tile 解码与金标准 C_ws 比对。
                // 注意: 多核(多S2块)时各核dump会重叠, 本模式仅用于单块形状。
                if (aivIdx == 0) {
                    LocalTensor<int32_t> dumpUb = outQueue_.AllocTensor<int32_t>();
                    DataCopy(dumpUb, scoreUb.template ReinterpretCast<int32_t>(), cuS2LenVecAlign);
                    PipeBarrier<PIPE_V>();
                    outQueue_.EnQue<int32_t>(dumpUb);
                    dumpUb = outQueue_.DeQue<int32_t>();
                    LIQServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset], dumpUb, cuS2LenVecAlign);
                    outQueue_.FreeTensor(dumpUb);
                    // 落址探针: 64 个斜坡值(1000..1063)写到 offset 512, host 侧核对实际落点
                    LocalTensor<int32_t> probeUb = outQueue_.AllocTensor<int32_t>();
                    ArithProgression<int32_t>(probeUb, 1000, 1, 64);
                    PipeBarrier<PIPE_V>();
                    outQueue_.EnQue<int32_t>(probeUb);
                    probeUb = outQueue_.DeQue<int32_t>();
                    LIQServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + 512], probeUb, 64);
                    outQueue_.FreeTensor(probeUb);
                }
                inQueue_.FreeTensor(mmInUb);
                return;
            }
            AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, cuS2Len);
            PipeBarrier<PIPE_V>();
            AscendC::Mul(scoreUb, scoreUb, kScaleUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            //---- ECHO-style prefetch: 对当前token的 scores 做 threshold 比较并触发 H2D ----
            PrefetchScoresInVec1(scoreUb, info, cuS2Len, cuBaseS2Idx);
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIQServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, scoreUb, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(mmInUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> tmpSortBuf = sortBuff[2 * cuS2LenVecAlign];
            LIQServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            LIQServiceVec::MergeSort(globalTopkUb_[ti * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK, sortBuff,
                                     cuS2LenVecAlign, tmpSortBuf);
            PipeBarrier<PIPE_V>();
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            // 中间结果保存
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;
            if (needCopyOutGm) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                ExtractIndex(idxULocal,
                             globalTopkUb_[ti * BASE_TOPK_VALUE_IDX_SIZE].template ReinterpretCast<uint32_t>(),
                             BASE_TOPK);
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[ti * BASE_TOPK_VALUE_IDX_SIZE], BASE_TOPK_VALUE_IDX_SIZE);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                LIQServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                       idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]
                // tLocal为块内绝对S1行, 两个AIV交错写入互不重叠
                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * BASE_TOPK_VALUE_IDX_SIZE +
                                   tLocal * 2 * BASE_TOPK_VALUE_IDX_SIZE;
                int64_t wsInfoOffset =
                    (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ + tLocal * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                // 写入头尾判断
                // [head, tail]
                // head: 与前面规约，与前后规约
                // tail: 与后面规约
                bool isTailReduce = blockS2StartIdx_ == 0;  // 一定是isLastTile
                // WS偏移规则 blockS2StartIdx_ != 0
                // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
                // 跟后面块做规约 写到1偏移  需要 + s1BaseSize_, BASE_TOPK*2
                if (isTailReduce) {  // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += BASE_TOPK_VALUE_IDX_SIZE;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIQServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIQServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[ti * BASE_TOPK_VALUE_IDX_SIZE],
                                       BASE_TOPK_VALUE_IDX_SIZE);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    // BNSD场景无效S1 输出-1
    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            for (int32_t s1Idx = info.actS1Size + aivIdx; s1Idx < constInfo_.qSeqSize; s1Idx += 2) {
                CleanInvalidOutput(info.indiceOutOffset + s1Idx * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            for (int32_t s1Idx = aivIdx; s1Idx < invalidS1Num2; s1Idx += 2) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1Idx) * constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIQT>
__aicore__ inline void LIQVector<LIQT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * BASE_TOPK_VALUE_IDX_SIZE +  // 2个AIV共同地址偏移
                   innerS1Idx * 2 * BASE_TOPK_VALUE_IDX_SIZE + BASE_TOPK_VALUE_IDX_SIZE;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(BASE_TOPK_VALUE_IDX_SIZE * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += BASE_TOPK_VALUE_IDX_SIZE;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + 6);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * BASE_TOPK_VALUE_IDX_SIZE +  // 2个AIV共同地址偏移
                       innerS1Idx * 2 * BASE_TOPK_VALUE_IDX_SIZE;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(BASE_TOPK_VALUE_IDX_SIZE * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += BASE_TOPK_VALUE_IDX_SIZE;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[BASE_TOPK_VALUE_IDX_SIZE];
                srcList.src3 = curValueIdxUb[2 * BASE_TOPK_VALUE_IDX_SIZE];
                srcList.src4 = curValueIdxUb[3 * BASE_TOPK_VALUE_IDX_SIZE];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, BASE_TOPK_VALUE_IDX_SIZE);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = BASE_TOPK_VALUE_IDX_SIZE;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src3 = curValueIdxUb[2 * BASE_TOPK_VALUE_IDX_SIZE];
            srcList.src4 = curValueIdxUb[3 * BASE_TOPK_VALUE_IDX_SIZE];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, BASE_TOPK_VALUE_IDX_SIZE);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        LIQServiceVec::CopyOut(indiceOutGm[outOffset], idxULocal1, constInfo_.sparseCount);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
}
}  // namespace LIQKernel
#endif
