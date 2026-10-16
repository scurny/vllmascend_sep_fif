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
 * \file lightning_indexer_quant_service_cube.h
 * \brief 950(arch35) cube: 仅计算Qk(int8*int8->int32 L0C), 结果经C310 Fixpipe
 *        直写配对AIV的UB(dualDstCtl按M/2拆分), 加权归约在AIV侧完成。
 */
#ifndef LIGHTNING_INDEXER_QUANT_SERVICE_CUBE_H
#define LIGHTNING_INDEXER_QUANT_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_quant_common_arch35.h"

namespace LIQKernel {
using namespace LIQCommon;
struct MmInfo {
    int64_t s2L0LoopId;
    int64_t s1gL0LoopId;
    int64_t s2L0RealSize;
    int64_t s2GmOffset;
};

template <typename LIQT>
class LIQMatmul {
public:
    using Q_T = typename LIQT::queryType;
    using K_T = typename LIQT::keyType;

    __aicore__ inline LIQMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                                const GlobalTensor<Q_T> &queryGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const LIQCommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true};  // isSetFMatrix isSetPadding;
    static constexpr uint64_t DOUBLE_BUF_NUM = 2;
    static constexpr uint64_t L0AB_BUF_NUM = 4;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QW_MTE1_MTE2_EVENT = EVENT_ID5;  // KEY_MTE1_MTE2_EVENT + DOUBLE_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;
    static constexpr uint32_t M_FIX_EVENT = EVENT_ID0;
    static constexpr uint32_t FIX_M_EVENT = EVENT_ID2;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t D_BASIC_BLOCK = 128;
    static constexpr uint64_t S1G_BASIC_BLOCK_L1 = 256;

    static constexpr uint64_t S1G_BASIC_BLOCK_L0 = 128;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 128;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK;
    static constexpr uint64_t L0AB_BUFFER_OFFSET_S8_16K = 16 * 1024;
    static constexpr uint64_t L0C_BUFFER_OFFSET = 64 * 256;

    // ---- 950 UB直通: qk结果缓冲(与AIV侧qkBuf_同布局, 融合核fixpipe跨核落址) ----
    // 每slot: [M/2=64行 x 128列] int32, 行距dstStride=128(连续); ping-pong共2 slot
    static constexpr uint64_t QK_SLOT_ELEMS = (S1G_BASIC_BLOCK_L0 / DOUBLE_BUF_NUM) * S2_BASIC_BLOCK_L0;
    static constexpr uint64_t QK_UB_BYTES = DOUBLE_BUF_NUM * QK_SLOT_ELEMS * sizeof(int32_t);
    static constexpr FixpipeConfig LIQ_CFG_ROW_MAJOR_UB = {
        CO2Layout::ROW_MAJOR,
        // ROW_MAJOR: 使能NZ2ND, 输出ND格式; true: 目的地址是UB
        true};

private:
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0RealSize);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize, uint64_t s1gL0RealSize);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, const LIQCommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const LIQCommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo &mmInfo, const LIQCommon::RunInfo &runInfo);
    __aicore__ inline void FixpQkToUB(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, uint32_t slot);
    __aicore__ inline void ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize);
    __aicore__ inline void ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                     const MmInfo &mmInfo, const LIQCommon::RunInfo &runInfo);
    __aicore__ inline void CalcMmInfo(MmInfo &mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt, const MmInfo &lastMmInfo,
                                      const LIQCommon::RunInfo &runInfo);
    static constexpr LI_LAYOUT Q_LAYOUT_T = LIQT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = LIQT::keyLayout;
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    // 950 UB直通: 必须是cube侧TPipe的第一个VECCALC缓冲, 与AIV侧qkBuf_地址对齐
    TBuf<TPosition::VECCALC> bufQkUb_;
    LocalTensor<int32_t> qkUb_;

    TBuf<TPosition::A2> bufL0A_;
    LocalTensor<Q_T> l0a_;
    TBuf<TPosition::B2> bufL0B_;
    LocalTensor<K_T> l0b_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t qwL1Mte2BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;

    ConstInfo constInfo_;
};

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::InitBuffers(TPipe *pipe)
{
    // 950 UB直通: 第一个(唯一)VECCALC缓冲, 与AIV侧qkBuf_基地址一致
    pipe->InitBuffer(bufQkUb_, QK_UB_BYTES);
    qkUb_ = bufQkUb_.Get<int32_t>();

    pipe->InitBuffer(bufQL1_, DOUBLE_BUF_NUM * S1G_BASIC_BLOCK_L1 * D_BASIC_BLOCK * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, DOUBLE_BUF_NUM * S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufL0A_, 64 * 1024);
    l0a_ = bufL0A_.Get<Q_T>();
    pipe->InitBuffer(bufL0B_, 64 * 1024);
    l0b_ = bufL0B_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, 128 * 1024);
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm,
                                                            const GlobalTensor<K_T> &keyGm,
                                                            const GlobalTensor<Q_T> &queryGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::CalcMmInfo(MmInfo &mmInfo, uint64_t loopIdx, uint64_t s1L0LoopCnt,
                                                   const MmInfo &lastMmInfo, const LIQCommon::RunInfo &runInfo)
{
    mmInfo.s2L0LoopId = loopIdx / s1L0LoopCnt;
    mmInfo.s1gL0LoopId = loopIdx % s1L0LoopCnt;

    if (mmInfo.s1gL0LoopId == 0) {
        mmInfo.s2GmOffset = mmInfo.s2L0LoopId * S2_BASIC_BLOCK_L0;
        mmInfo.s2L0RealSize = mmInfo.s2GmOffset + S2_BASIC_BLOCK_L0 > runInfo.actualSingleProcessSInnerSize
                                  ? runInfo.actualSingleProcessSInnerSize - mmInfo.s2GmOffset
                                  : S2_BASIC_BLOCK_L0;
    } else {
        mmInfo.s2L0RealSize = lastMmInfo.s2L0RealSize;
    }
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::ComputeMm1(const LIQCommon::RunInfo &runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        QueryNd2Nz(runInfo.actMBaseSize, runInfo);  // 256 * 128 // L1BasicBlock
    }
    int64_t s2L0LoopCnt = CeilDiv(runInfo.actualSingleProcessSInnerSize, S2_BASIC_BLOCK_L0);  // 2048取128
    int64_t s1L0LoopCnt = CeilDiv(runInfo.actMBaseSize, S1G_BASIC_BLOCK_L0);                  // 256取128
    int64_t s1gL1Offset[2] = {0, static_cast<int64_t>(S1G_BASIC_BLOCK_L0)};
    int64_t s1gL0RealSize[2] = {s1L0LoopCnt > 1 ? static_cast<int64_t>(S1G_BASIC_BLOCK_L0) : runInfo.actMBaseSize,
                                runInfo.actMBaseSize - s1gL1Offset[1]};
    MmInfo mmInfo;
    for (int64_t loopIdx = 0; loopIdx < s2L0LoopCnt * s1L0LoopCnt; loopIdx++) {
        CalcMmInfo(mmInfo, loopIdx, s1L0LoopCnt, mmInfo, runInfo);
        ProcessQk(s1gL0RealSize[mmInfo.s1gL0LoopId % s1L0LoopCnt],
                  s1gL1Offset[mmInfo.s1gL0LoopId % s1L0LoopCnt], s1L0LoopCnt, mmInfo, runInfo);
    }

    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM);
        qwL1Mte2BufIdx_++;
    }
}

// blkNum, blkSize, N2, D
template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset,
                                                      const LIQCommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0;
    while (s2L1Offset < s2L1RealSize) {
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize;
        uint64_t keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                    constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
                                s2BlkOffset * constInfo_.headDim;
        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset;
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset
                                                                            : s2Mte2Size;
        Nd2NzParams nd2nzPara;
        nd2nzPara.ndNum = 1;
        nd2nzPara.nValue = s2Mte2Size;  // 行数
        nd2nzPara.dValue = constInfo_.headDim;
        nd2nzPara.srcDValue = constInfo_.headDim;
        nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
        nd2nzPara.dstNzNStride = 1;
        nd2nzPara.srcNdMatrixStride = 0;
        nd2nzPara.dstNzMatrixStride = 0;
        DataCopy(keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * S8_BLOCK_CUBE],
                 keyGm_[keyGmOffset], nd2nzPara);

        s2L1Offset += s2Mte2Size;
    }
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::KeyNd2Nz(uint64_t s2L1RealSize, const MmInfo &mmInfo,
                                                 const LIQCommon::RunInfo &runInfo)
{
    uint64_t dStride = constInfo_.headDim;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::BSND || K_LAYOUT_T == LI_LAYOUT::TND) {
        dStride = constInfo_.headDim * constInfo_.kHeadNum; // constInfo_.kHeadNum
    }
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize;  // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = dStride;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + mmInfo.s2GmOffset * constInfo_.headDim], nd2nzPara);
}

// batch, s1, n2, g, d
template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::QueryNd2Nz(uint64_t s1gL1RealSize, const LIQCommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize;  // 行数
    nd2nzPara.dValue = constInfo_.headDim;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);  // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET], queryGm_[runInfo.tensorQueryOffset],
             nd2nzPara);
}

// s1g, d
template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::LoadQueryToL0a(uint64_t s1gL1Offset, uint64_t s1gL1RealSize,
                                                       uint64_t s1gL0RealSize)
{
    LoadData3DParamsV2<Q_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = CeilDiv(s1gL1RealSize, BLOCK_CUBE);  // Hin=M1=8
    loadData3DParams.l1W = BLOCK_CUBE;                          // Win=M0
    loadData3DParams.channelSize = constInfo_.headDim;          // Cin=K

    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255;  // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = CeilAlign(s1gL0RealSize, BLOCK_CUBE);  // M height维度目的
    loadData3DParams.kExtension = constInfo_.headDim;                    // K   width维度目的
    loadData3DParams.mStartPt = s1gL1Offset;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;

    LoadData<Q_T, LOAD3DV2_CONFIG>(l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
                                   queryL1_[(qwL1Mte2BufIdx_ % DOUBLE_BUF_NUM) * QUERY_BUFFER_OFFSET],
                                   loadData3DParams);
}

// s2, d -> 128,128
template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::LoadKeyToL0b(uint64_t s2L0RealSize)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = CeilDiv(s2L0RealSize, BLOCK_CUBE) * CeilDiv(constInfo_.headDim, S8_BLOCK_CUBE);
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
             keyL1_[(keyL1BufIdx_ % DOUBLE_BUF_NUM) * KEY_BUFFER_OFFSET], loadData2DParams);
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::ComputeQk(uint64_t s1gL0RealSize, uint64_t s2L0RealSize)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);

    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, BLOCK_CUBE);
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    Mmad(cL0_[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET],
         l0a_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K],
         l0b_[(l0BufIdx_ % L0AB_BUF_NUM) * L0AB_BUFFER_OFFSET_S8_16K], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::ProcessQk(uint64_t s1gL0RealSize, uint64_t s1gL1Offset, uint64_t s1L0LoopCnt,
                                                  const MmInfo &mmInfo, const LIQCommon::RunInfo &runInfo)
{
    if (mmInfo.s1gL0LoopId == 0) {
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % DOUBLE_BUF_NUM);
        if constexpr (K_LAYOUT_T == LI_LAYOUT::PA_BSND) {
            KeyNd2NzForPA(mmInfo.s2L0RealSize, runInfo.s2Idx * constInfo_.s2BaseSize + mmInfo.s2GmOffset, runInfo);
        } else {
            KeyNd2Nz(mmInfo.s2L0RealSize, mmInfo, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
    }

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);
    LoadQueryToL0a(s1gL1Offset, runInfo.actMBaseSize, s1gL0RealSize);
    LoadKeyToL0b(mmInfo.s2L0RealSize);

    if (mmInfo.s1gL0LoopId + 1 >= s1L0LoopCnt) {
        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % DOUBLE_BUF_NUM);
        keyL1BufIdx_++;
    }

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    ComputeQk(s1gL0RealSize, mmInfo.s2L0RealSize);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0AB_BUF_NUM);

    // 950 UB直通: 等两个AIV消费完该slot上一轮数据, 再fixpipe写入
    uint32_t slot = static_cast<uint32_t>(mmInfo.s2L0LoopId * s1L0LoopCnt + mmInfo.s1gL0LoopId) % DOUBLE_BUF_NUM;
    CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(
        LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + slot * 2 + 0);
    CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(
        LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + slot * 2 + 1);
    FixpQkToUB(s1gL0RealSize, mmInfo.s2L0RealSize, slot);
    CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(
        LIQCommon::ConstInfo::SYNC_CV_EVENT_BASE + slot * 2 + 0);
    CrossCoreSetFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(
        LIQCommon::ConstInfo::SYNC_CV_EVENT_BASE + slot * 2 + 1);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + l0cBufIdx_ % DOUBLE_BUF_NUM);
    l0BufIdx_++;
    l0cBufIdx_++;
}

// L0C(int32) -> 配对AIV的UB. mSize恒128(尾块M<128时多余行为L0C残影, AIV按token有效性跳过);
// dualDstCtl=1: 前64行进AIV0的UB, 后64行进AIV1的UB (M/2按token边界对齐: 64行=1个token的64个head)
template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::FixpQkToUB(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, uint32_t slot)
{
    SetFlag<HardEvent::M_FIX>(M_FIX_EVENT);
    WaitFlag<HardEvent::M_FIX>(M_FIX_EVENT);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;
    fixpipeParams.nSize = (s2L0RealSize + 7) >> 3 << 3;  // 32B对齐
    fixpipeParams.mSize = S1G_BASIC_BLOCK_L0;            // 恒128, 保证dualDst拆分落在token边界
    fixpipeParams.srcStride = CeilAlign(s1gL0RealSize, BLOCK_CUBE);  // L0C NZ相邻大Z间距, 单位16*sizeof(T)
    fixpipeParams.dstStride = S2_BASIC_BLOCK_L0;         // UB行距(元素), 连续排布
    fixpipeParams.dualDstCtl = 1;                        // M/2*N写入每个AIV的UB
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<int32_t, int32_t, LIQ_CFG_ROW_MAJOR_UB>(
        qkUb_[slot * QK_SLOT_ELEMS], cL0_[(l0cBufIdx_ % DOUBLE_BUF_NUM) * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    SetFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);
}

template <typename LIQT>
__aicore__ inline void LIQMatmul<LIQT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QW_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 2);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 3);

    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 0);
    WaitFlag<HardEvent::FIX_M>(FIX_M_EVENT + 1);

    // 排空AIV最后一轮VC置位 (每事件恰余1次)
    CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + 0);
    CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + 1);
    CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + 2);
    CrossCoreWaitFlag<LIQCommon::ConstInfo::FIA_SYNC_MODE4, PIPE_FIX>(LIQCommon::ConstInfo::SYNC_VC_EVENT_BASE + 3);
}
}  // namespace LIQKernel
#endif
