/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
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
#ifndef LIGHTING_INDEXER_QUANT_VLLM_TORCH_ADPT_H
#define LIGHTING_INDEXER_QUANT_VLLM_TORCH_ADPT_H
#include <cstdio>
namespace vllm_ascend {

at::Tensor npu_lightning_indexer_quant(
    const at::Tensor &query, const at::Tensor &key, const at::Tensor &weights,
    const at::Tensor &query_dequant_scale, const at::Tensor &key_dequant_scale,
    const c10::optional<at::Tensor> &actual_seq_lengths_query,
    const c10::optional<at::Tensor> &actual_seq_lengths_key,
    const c10::optional<at::Tensor> &block_table, int64_t query_quant_mode, int64_t key_quant_mode,
    c10::string_view layout_query, c10::string_view layout_key, int64_t sparse_count, int64_t sparse_mode,
    // ---- ECHO-style prefetch ----
    const c10::optional<at::Tensor> &token_gva_table,
    const c10::optional<at::Tensor> &prefetch_staging_buf,
    const c10::optional<at::Tensor> &prefetch_slot_counter,
    const c10::optional<at::Tensor> &token_to_prefetch_slot,
    double ema_threshold, int64_t token_size_bytes_k,
    const c10::optional<at::Tensor> &sparse_values_out)
{
    auto check_prefetch_tensor = [&](const c10::optional<at::Tensor> &tensor,
                                     const char *name, bool writable) {
        if (!tensor.has_value()) {
            return;
        }
        TORCH_CHECK(tensor->defined(), name, " must be a defined tensor");
        TORCH_CHECK(tensor->device() == query.device(), name, " must be on the same device as query");
        TORCH_CHECK(!writable || tensor->is_contiguous(), name, " must be contiguous");
    };
    check_prefetch_tensor(token_gva_table, "token_gva_table", false);
    check_prefetch_tensor(prefetch_staging_buf, "prefetch_staging_buf", true);
    check_prefetch_tensor(prefetch_slot_counter, "prefetch_slot_counter", false);
    check_prefetch_tensor(token_to_prefetch_slot, "token_to_prefetch_slot", true);

    fprintf(stderr, "[LIQ_ADPT] enter npu_lightning_indexer_quant (adpt built %s %s)\n", __DATE__, __TIME__);
    fflush(stderr);
    std::string query_layout_str = std::string(layout_query);
    std::string key_layout_str = std::string(layout_key);

    const int SIZE = 8;
    const int DIM_0 = 0;
    const int DIM_1 = 1;
    const int DIM_2 = 2;
    const int DIM_3 = 3;

    at::SmallVector<int64_t, SIZE> output_size;
    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
            "than 0, but shape[", i, "] is ", query.size(i));
    }
    for (size_t i = 0; i < key.sizes().size(); i++) {
        TORCH_CHECK(key.size(i) > 0, "All values within key's shape should be greater "
            "than 0, but shape[", i, "] is ", key.size(i));
    }
    TORCH_CHECK(sparse_count > 0, "sparse count should be greater than 0, but now is ", sparse_count);
    int64_t keyHeadNum = (key_layout_str == "TND")? key.size(DIM_1) : key.size(DIM_2);
    if (query_layout_str == "BSND") {
        output_size = {query.size(DIM_0), query.size(DIM_1), keyHeadNum, sparse_count};
    } else {
        output_size = {query.size(DIM_0), keyHeadNum, sparse_count};
    }
    at::Tensor lightning_indexer_quant_output = at::empty(output_size, query.options().dtype(at::kInt));
    if (sparse_values_out.has_value()) {
        TORCH_CHECK(sparse_values_out->defined(), "sparse_values_out must be a defined tensor");
        TORCH_CHECK(sparse_values_out->device() == query.device(),
                    "sparse_values_out must be on the same device as query");
        TORCH_CHECK(sparse_values_out->scalar_type() == at::kFloat,
                    "sparse_values_out must have dtype float32");
        TORCH_CHECK(sparse_values_out->is_contiguous(), "sparse_values_out must be contiguous");
        TORCH_CHECK(sparse_values_out->sizes() == at::IntArrayRef(output_size),
                    "sparse_values_out must have the same shape as sparse_indices");
    }

    // convert str
    char *query_layout_ptr = const_cast<char *>(query_layout_str.c_str());
    char *key_layout_ptr = const_cast<char *>(key_layout_str.c_str());

    //---- ECHO-style prefetch: 局部变量，避免 rvalue 绑定非 const 引用 ----
    // NOTE: def 里 .Float() 声明的 attr, aclnn 生成签名按仓库惯例收 double;
    // 传 float 会因寄存器位宽被按 double 读成 subnormal -> 0.0, 故必须保持 double。
    double ema_threshold_val = ema_threshold;
    int64_t token_size_bytes_k_val = token_size_bytes_k;
    // NOTE: EXEC_NPU_CMD 按位置直接调用生成的 aclnnLightningIndexerQuantGetWorkspaceSize,
    // 参数顺序必须与 op_host/lightning_indexer_quant_def.cpp 中的声明顺序完全一致:
    // 先全部 inputs(query...block_table + 4 个 prefetch 输入 + sparse_values_out), 再 attrs, 最后 output。
    fprintf(stderr,
            "[LIQ_ADPT] before EXEC_NPU_CMD: q=[%ld,%ld,%ld] k=[%ld,%ld,%ld,%ld] layout=%s/%s "
            "sparse_count=%ld sparse_mode=%ld ema=%f tskb=%ld "
            "has_gva=%d has_staging=%d has_counter=%d has_slot=%d\n",
            (long)query.size(0), query.dim() > 1 ? (long)query.size(1) : -1,
            query.dim() > 2 ? (long)query.size(2) : -1,
            (long)key.size(0), (long)key.size(1), (long)key.size(2),
            key.dim() > 3 ? (long)key.size(3) : -1,
            query_layout_str.c_str(), key_layout_str.c_str(),
            (long)sparse_count, (long)sparse_mode, ema_threshold_val,
            (long)token_size_bytes_k_val,
            token_gva_table.has_value() ? 1 : 0,
            prefetch_staging_buf.has_value() ? 1 : 0,
            prefetch_slot_counter.has_value() ? 1 : 0,
            token_to_prefetch_slot.has_value() ? 1 : 0);
    fflush(stderr);
    EXEC_NPU_CMD(aclnnLightningIndexerQuant,
    query,
    key,
    weights,
    query_dequant_scale,
    key_dequant_scale,
    actual_seq_lengths_query,
    actual_seq_lengths_key,
    block_table,
    token_gva_table,
    prefetch_staging_buf,
    prefetch_slot_counter,
    token_to_prefetch_slot,
    sparse_values_out,
    query_quant_mode,
    key_quant_mode,
    query_layout_ptr,
    key_layout_ptr,
    sparse_count,
    sparse_mode,
    ema_threshold_val,
    token_size_bytes_k_val,
    lightning_indexer_quant_output
    );
    fprintf(stderr, "[LIQ_ADPT] after EXEC_NPU_CMD: out=[%ld,%ld,%ld]\n",
            (long)lightning_indexer_quant_output.size(0),
            lightning_indexer_quant_output.dim() > 1 ? (long)lightning_indexer_quant_output.size(1) : -1,
            lightning_indexer_quant_output.dim() > 2 ? (long)lightning_indexer_quant_output.size(2) : -1);
    fflush(stderr);

    return lightning_indexer_quant_output;

}
}
#endif
