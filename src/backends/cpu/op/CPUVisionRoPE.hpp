
#ifndef MLLM_CPUVISIONROPE_H
#define MLLM_CPUVISIONROPE_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUVisionRoPE final : public Op {
public:
    CPUVisionRoPE(Backend *bn, string opName, int dim, int spatial_merge_size, int threadCount);
    virtual ~CPUVisionRoPE() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    // virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
    int spatial_merge_size_;
    int dim_;
    
    std::vector<float> inv_freq;  // 存储计算结果
    float theta = 10000.0f;       // 默认theta值，与Transformer常用的值一致

    // vector<vector<int>> pos_ids;
    int max_grid_size;

    void compute_inv_freq(int dim);
    std::vector<std::vector<float>> rotary_pos_emb_forward(int seqlen);
    vector<vector<int>> rot_pos_emb(vector<vector<float>> grid_thw, int spatial_merge_size);
    void compute_rotary_pos_embd(
        const vector<vector<float>>& rotary_pos_emb_full,
        const vector<vector<int>>& pos_ids,
        shared_ptr<Tensor> output
    );
};

class CPUVisionRoPECreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        int dims = (int)op_param["dim"];
        int spatial_merge_size = (int)op_param["spatial_merge_size"];
        return new CPUVisionRoPE(bn, name, dims, spatial_merge_size, threadCount);
    }
};


/*
class VisionRotaryEmbedding {
private:
    std::vector<float> inv_freq;
    float theta;
    const int dim;
    
public:
    // 构造函数
    VisionRotaryEmbedding(int dim, float theta = 10000.0f) : dim(dim), theta(theta) 
    {
        if (dim <= 0 || dim % 2 != 0) {
            throw std::invalid_argument("Dimension must be positive even number");
        }
        compute_inv_freq();
    }
    
    // 计算逆频率向量
    void compute_inv_freq() {
        const int half_dim = dim / 2;
        inv_freq.resize(half_dim);
        #pragma omp parallel for num_threads(4)
        for (int i = 0; i < half_dim; ++i) {
            const float exponent = (2.0f * i) / static_cast<float>(dim);
            inv_freq[i] = 1.0f / std::pow(theta, exponent);
        }
    }
    
    // 前向计算（生成频率矩阵）
    std::vector<std::vector<float>> rotary_pos_emb_forward(int seqlen) {
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
    
        // 访问器方法
    const std::vector<float>& get_inv_freq() const { return inv_freq; }
    float get_theta() const { return theta; }
    int get_dim() const { return dim; }
};
*/

} // namespace mllm

#endif // MLLM_CPUVISIONROPE_H
