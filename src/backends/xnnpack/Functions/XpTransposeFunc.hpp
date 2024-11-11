/**
 * @file XpTransposeFunc.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-16
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Backend.hpp"
#include "xnnpack/XpInterface.hpp"
namespace mllm::xnnpack {

class XpTransposeFunction : public TensorFunction, public XpTensorDefineInterface<XpTransposeFunction> {
public:
    void setup(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override;

    void execute(vector<Tensor *> outputs, vector<Tensor *> inputs, vector<float> args) override;
};

} // namespace mllm::xnnpack