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
 * \file lightning_indexer_quant_def.cpp
 * \brief
 */
#include <cstdint>

#include "register/op_def_registry.h"

namespace ops {
class LightningIndexerQuant : public OpDef {
public:
    explicit LightningIndexerQuant(const char *name) : OpDef(name)
    {
        this->Input("query")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("weights")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("query_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("key_dequant_scale")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_query")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("actual_seq_lengths_key")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        this->Input("block_table")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // ---- ECHO-style prefetch inputs (OPTIONAL) ----
        this->Input("token_gva_table")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        this->Input("prefetch_staging_buf")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT8})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        this->Input("prefetch_slot_counter")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        this->Input("token_to_prefetch_slot")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        this->Output ("sparse_indices").ParamType (REQUIRED).DataType ({ge::DT_INT32}).FormatList ({ge::FORMAT_ND});
        this->Attr("query_quant_mode").AttrType(REQUIRED).Int(0);  // 0: 默认值，per-token-head
        this->Attr("key_quant_mode").AttrType(REQUIRED).Int(0);    // 0: 默认值，per-token-head
        this->Attr("layout_query").AttrType(OPTIONAL).String("BSND");
        this->Attr("layout_key").AttrType(OPTIONAL).String("PA_BSND");
        this->Attr("sparse_count").AttrType(OPTIONAL).Int(2048);  // 2048: 默认值，筛选前2048
        this->Attr("sparse_mode").AttrType(OPTIONAL).Int(3);      // 3: 默认值，只计算下三角
        // ---- ECHO-style prefetch attr ----
        this->Attr ("ema_threshold").AttrType (OPTIONAL).Float (0.0f);    // EMA 预测阈值，0.0 = 不触发 prefetch
        this->Attr ("token_size_bytes_k").AttrType (OPTIONAL).Int (0);    // 单 token K 字节数
        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
            .ExtendCfgInfo("jitCompile.flag", "static_false,dynamic_false");
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
        //---- A5 (ascend950) 配置 ----
        OpAICoreConfig aicore_config_95;
        aicore_config_95.Input("query")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8,ge::DT_INT8,ge::DT_INT8,ge::DT_INT8,ge::DT_INT8})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("key")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8,ge::DT_INT8,ge::DT_INT8,ge::DT_INT8,ge::DT_INT8})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("weights")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("query_dequant_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("key_dequant_scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("actual_seq_lengths_query")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("actual_seq_lengths_key")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("block_table")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        // ---- ECHO-style prefetch inputs ----
        aicore_config_95.Input("token_gva_table")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT64,ge::DT_UINT64,ge::DT_UINT64,ge::DT_UINT64,ge::DT_UINT64})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("prefetch_staging_buf")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_UINT8,ge::DT_UINT8,ge::DT_UINT8,ge::DT_UINT8,ge::DT_UINT8})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("prefetch_slot_counter")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Input("token_to_prefetch_slot")
        .ParamType(OPTIONAL)
        .DataType({ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32})
        .FormatList({ge::FORMAT_ND})
        .AutoContiguous();
        aicore_config_95.Output("sparse_indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32,ge::DT_INT32})
        .FormatList({ge::FORMAT_ND});
        aicore_config_95.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true);
        this->AICore().AddConfig("ascend950", aicore_config_95);
    }
};
OP_ADD(LightningIndexerQuant);
}  // namespace ops