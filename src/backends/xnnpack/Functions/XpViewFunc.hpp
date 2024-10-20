/**
 * @file XpViewFunc.hpp
 * @author your name (you@domain.com)
 * @version 0.1
 * @date 2024-10-20
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Tensor.hpp"
#include "Types.hpp"
#include "Backend.hpp"

namespace mllm::xnnpack {

class XpViewFunction : public TensorFunction {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override;
    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override;
};
} // namespace mllm::xnnpack