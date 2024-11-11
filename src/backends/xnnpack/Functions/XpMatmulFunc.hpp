/**
 * @file XpMatmulFunc.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-10-23
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Backend.hpp"
#include "xnnpack/XpInterface.hpp"

namespace mllm::xnnpack {

class XpMatmulFunction : public TensorFunction, public XpTensorDefineInterface<XpMatmulFunction> {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override;

    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override;
};
} // namespace mllm::xnnpack