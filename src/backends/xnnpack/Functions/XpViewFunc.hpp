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
#include "backends/xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {

class XpViewFunction : public TensorFunction, XpTensorDefineInterface<XpViewFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};
} // namespace mllm::xnnpack