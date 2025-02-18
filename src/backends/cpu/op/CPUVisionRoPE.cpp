#include "CPUVisionRoPE.hpp"
#include <cassert>
#include <vector>
#include <algorithm>
#include "CPUBackend.hpp"
#include "compute/SIMDMemory.hpp"


using namespace std;

namespace mllm {
CPUVisionRoPE::CPUVisionRoPE(Backend *bn, string opName, int dim, int spatial_merge_size, int threadCount): thread_count(threadCount),
    Op(bn, opName){
    dim_ = dim;
    spatial_merge_size_ = spatial_merge_size;
    compute_inv_freq(dim);
}
// 计算并填充inv_freq向量的方法
void CPUVisionRoPE::compute_inv_freq(int dim) {
    const int half_dim = dim / 2;
    inv_freq.clear();
    inv_freq.resize(half_dim);
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < half_dim; ++i) {
        const float exponent = (2.0f * i) / static_cast<float>(dim);
        inv_freq[i] = 1.0f / std::pow(theta, exponent);
    }
}
std::vector<std::vector<float>> CPUVisionRoPE::rotary_pos_emb_forward(int seqlen) {
    if (seqlen <= 0) {
        throw std::invalid_argument("seqlen must be positive");
    }
    // 生成序列 [0, 1, ..., seqlen-1]
    std::vector<float> seq(seqlen);
    #pragma omp parallel for num_threads(4)
    for (int i = 0; i < seqlen; ++i) {
        seq[i] = static_cast<float>(i);
    }
    // 预分配结果矩阵
    std::vector<std::vector<float>> freqs(
        seqlen, 
        std::vector<float>(inv_freq.size())
    );
    // 并行计算外积
    #pragma omp parallel for num_threads(4) schedule(dynamic)
    for (size_t i = 0; i < seq.size(); ++i) {
        const float seq_val = seq[i];
        auto& row = freqs[i];
        for (size_t j = 0; j < inv_freq.size(); ++j) {
            row[j] = seq_val * inv_freq[j];
        }
    }
    return freqs;
}

vector<vector<int>> CPUVisionRoPE::rot_pos_emb(vector<vector<float>> grid_thw, int spatial_merge_size) {
    vector<vector<int>> pos_ids;
    // int max_grid_size = 0;
    // 遍历每个时空网格配置
    for (auto& row : grid_thw) {
        int t = static_cast<int>(row[0]);
        int h = static_cast<int>(row[1]);
        int w = static_cast<int>(row[2]);
        // 更新最大空间网格尺寸
        max_grid_size = max({max_grid_size, h, w});
        // 计算分块参数
        int num_h_blocks = h / spatial_merge_size;
        int num_w_blocks = w / spatial_merge_size;
        int total_blocks = num_h_blocks * num_w_blocks;
        const int block_area = spatial_merge_size * spatial_merge_size;
        // 预分配内存
        vector<int> flatten_hpos(total_blocks * block_area);
        vector<int> flatten_wpos(total_blocks * block_area);
        // 并行生成坐标序列
        #pragma omp parallel for num_threads(thread_count) schedule(static)
        for (int block_idx = 0; block_idx < total_blocks; ++block_idx) {
            const int i_h = block_idx / num_w_blocks;
            const int i_w = block_idx % num_w_blocks;
            const int start_idx = block_idx * block_area;
            // 生成块内坐标
            for (int j_h = 0; j_h < spatial_merge_size; ++j_h) {
                for (int j_w = 0; j_w < spatial_merge_size; ++j_w) {
                    const int pos = start_idx + j_h * spatial_merge_size + j_w;
                    flatten_hpos[pos] = i_h * spatial_merge_size + j_h;
                    flatten_wpos[pos] = i_w * spatial_merge_size + j_w;
                }
            }
        }
        // 创建坐标对并重复时间维度
        vector<vector<int>> current_pos;
        current_pos.reserve(flatten_hpos.size());
        for (size_t i = 0; i < flatten_hpos.size(); ++i) {
            current_pos.push_back({flatten_hpos[i], flatten_wpos[i]});
        }
        // 扩展时间维度
        for (int i = 0; i < t; ++i) {
            pos_ids.insert(pos_ids.end(), current_pos.begin(), current_pos.end());
        }
    }
    // return {pos_ids, max_grid_size};
    return pos_ids;
}

void CPUVisionRoPE::compute_rotary_pos_embd(
    const vector<vector<float>>& rotary_pos_emb_full,
    const vector<vector<int>>& pos_ids,
    shared_ptr<Tensor> output) 
{
    // 输入验证
    if (rotary_pos_emb_full.empty() || pos_ids.empty()) {
        throw invalid_argument("Input containers must not be empty");
    }
    const size_t num_positions = rotary_pos_emb_full.size();
    const size_t dim = rotary_pos_emb_full[0].size();
    const size_t batch_size = pos_ids.size();
    // 验证嵌入维度一致性
    for (const auto& emb : rotary_pos_emb_full) {
        if (emb.size() != dim) {
            throw invalid_argument("Inconsistent embedding dimensions");
        }
    }
    // 验证位置ID有效性
    const size_t seq_len = pos_ids[0].size();
    for (const auto& ids : pos_ids) {
        if (ids.size() != seq_len) {
            throw invalid_argument("Varied sequence lengths not supported");
        }
        for (int idx : ids) {
            if (idx < 0 || idx >= num_positions) {
                throw out_of_range("Position index out of bounds");
            }
        }
    }
    // 准备输出Tensor
    const size_t flattened_size = seq_len * dim;
    assert(flattened_size == output->dimension()); // output->resize(1, 1, batch_size, flattened_size);
    // 并行处理主循环
    #pragma omp parallel for num_threads(CPUBackend::cpu_threads)
    for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        float* batch_ptr = output->ptrAt<float>(0, 0, batch_idx, 0);
        size_t offset = 0;
        for (const int pos_idx : pos_ids[batch_idx]) {
            const auto& emb = rotary_pos_emb_full[pos_idx];
            simd_memcpy(batch_ptr + offset, emb.data(), dim);
            offset += dim;
        }
    }
}

/*
vector<vector<float>> compute_rotary_positional_embeddings(
    const vector<vector<float>>& rotary_pos_emb_full,
    const vector<vector<int>>& pos_ids,
    shared_ptr<Tensor> output) 
{
    // 输入验证
    if (rotary_pos_emb_full.empty() || pos_ids.empty()) {
        return {};
    }
    const size_t num_positions = rotary_pos_emb_full.size();
    const size_t dim = rotary_pos_emb_full[0].size();
    const size_t batch_size = pos_ids.size();
    // 验证所有位置的维度一致性
    for (const auto& emb : rotary_pos_emb_full) {
        if (emb.size() != dim) {
            throw invalid_argument("All positional embeddings must have the same dimension");
        }
    }
    vector<vector<float>> result(batch_size);
    // 并行处理每个样本
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        const auto& positions = pos_ids[i];
        vector<float> flattened;
        flattened.reserve(positions.size() * dim);
        for (const int idx : positions) {
            // 边界检查
            if (idx < 0 || idx >= num_positions) {
                throw out_of_range("Position index out of range");
            }
            // 获取对应的位置嵌入
            const auto& emb = rotary_pos_emb_full[idx];
            // 使用SIMD加速的内存复制
            const size_t prev_size = flattened.size();
            flattened.resize(prev_size + dim);
            simd_memcpy(flattened.data() + prev_size, emb.data(), dim);
        }
        // 移动语义优化内存分配
        result[i] = std::move(flattened);
    }
    return result;
}
*/

ErrorCode CPUVisionRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int grid_t = (int)inputs[0]->dataAt<float>(0,0,0,0);
    int grid_h = (int)inputs[0]->dataAt<float>(0,0,0,1);
    int grid_w = (int)inputs[0]->dataAt<float>(0,0,0,2);
    outputs[0]->reshape(1, 1, grid_t*grid_h*grid_w, 2*(dim_/2));
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUVisionRoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    vector<vector<float>> grid_thw={{inputs[0]->dataAt<float>(0,0,0,0),
                                        inputs[0]->dataAt<float>(0,0,0,1),
                                        inputs[0]->dataAt<float>(0,0,0,2)}};
    auto pos_ids = rot_pos_emb(grid_thw, spatial_merge_size_); //get pos_ids and max_grid_size
    auto rotary_pos_emb_full = rotary_pos_emb_forward(max_grid_size);
    compute_rotary_pos_embd(rotary_pos_emb_full, pos_ids, outputs[0]);
    return Op::execute(inputs, outputs);
}

} // namespace mllm