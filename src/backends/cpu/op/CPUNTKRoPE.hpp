/**
 * @file CPUNTKRoPE.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-01-08
 *
 * @copyright Copyright (c) 2025
 *
 */
#pragma once

#include "Op.hpp"
#include "../CPUBackend.hpp"

// 1. Scaling factor
// \text{scale} = \frac{\text{max\_position\_embeddings}}{\text{original\_max\_position\_embeddings}}
// \text{scaling\_factor} = \sqrt{1 + \frac{\log(\text{scale})}{\log(\text{original\_max\_position\_embeddings})}}

// 2. Frequency Calculation
// t = [0, 1, 2, \dots, \text{seq\_len} - 1]
// \text{ext\_factors} =
// \begin{cases}
// \text{long\_factor} & \text{if } \text{seq\_len} > \text{original\_max\_position\_embeddings} \\
// \text{short\_factor} & \text{otherwise}
// \end{cases}
// \text{freqs} = \left(t \cdot \frac{1}{\text{ext\_factors}}\right) \otimes \text{inv\_freq}

// 3. Rotary Position Embedding
// \text{emb} = [\text{freqs}, \text{freqs}]
// \text{cos\_cached} = \cos(\text{emb}) \cdot \text{scaling\_factor}
// \text{sin\_cached} = \sin(\text{emb}) \cdot \text{scaling\_factor}

// 4. all
// \text{RoPE}(x, t) =
// \begin{bmatrix}
// \cos(\theta_t) & -\sin(\theta_t) \\
// \sin(\theta_t) & \cos(\theta_t)
// \end{bmatrix}
// \cdot x
//
// \theta_t = t \cdot \frac{1}{\text{ext\_factors}} \cdot \text{inv\_freq}

namespace mllm {

class CPUNTKRoPE final : public Op {
public:
    CPUNTKRoPE(Backend *bn, string op_name, int pose_type, int thread_count);
    CPUNTKRoPE(Backend *bn, string op_name, int pose_type, float rope_theta,
               const std::vector<float> &long_factor,
               const std::vector<float> &short_factor,
               int original_max_position_embeddings,
               int max_position_embeddings,
               int thread_count);

    ~CPUNTKRoPE() override = default;
    ErrorCode reshape(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) override;

    // FIXME: Typo here !!! Abstract
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode execute(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) override;
    ErrorCode free(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs) override;
    ErrorCode doExecute(std::vector<std::shared_ptr<Tensor>> inputs, std::vector<std::shared_ptr<Tensor>> outputs);

private:
    static int in_shape_old;
    static std::vector<std::vector<float>> emb_sin_;
    static std::vector<std::vector<float>> emb_cos_;
    std::vector<float> long_factor_;
    std::vector<float> short_factor_;
    int pose_type_ = 4;
    int thread_count_ = 4;
    int h_cnt_ = 0;
    float rope_theta_ = 1e-4f;
    int max_position_embeddings_ = 32768;
    int original_max_position_embeddings_ = 32768;
    int in_shape = -1;

    void
    clearCache() override {
        h_cnt_ = 0;
    }
};

class CPUNTKRoPECreator : public CPUBackend::Creator {
public:
    // FIXME: OpParam is copied.
    // FIXME: name is copied, may optimized to move by compiler.
    Op *create(OpParam op_param, Backend *bn, string name, int thread_count) const override {
        int pose_type = static_cast<int>(op_param["pose_type"]);
        float rope_theta = op_param["theta"];
        int max_position_embeddings = static_cast<int>(op_param["max_position_embeddings"]);

        int long_factor_n = static_cast<int>(op_param["long_factor_n"]);
        int short_factor_n = static_cast<int>(op_param["short_factor_n"]);
        std::vector<float> long_factor;
        std::vector<float> short_factor;

        // FIXME: the way we pass vector to backend is inefficient.
        for (int _i_long_factor_n = 0; _i_long_factor_n < long_factor_n; _i_long_factor_n++) {
            long_factor.push_back(op_param["long_factor_" + std::to_string(_i_long_factor_n)]);
        }

        for (int _i_short_factor_n = 0; _i_short_factor_n < short_factor_n; _i_short_factor_n++) {
            short_factor.push_back(op_param["short_factor_" + std::to_string(_i_short_factor_n)]);
        }

        int original_max_position_embeddings = static_cast<int>(op_param["original_max_position_embeddings"]);

        return new CPUNTKRoPE(bn, name, pose_type, rope_theta, long_factor, short_factor,
                              original_max_position_embeddings, max_position_embeddings, thread_count);
    }
};

} // namespace mllm