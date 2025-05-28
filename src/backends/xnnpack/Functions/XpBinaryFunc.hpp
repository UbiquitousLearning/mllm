/**
 * @file XpBinaryFunc.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-10
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Backend.hpp"
#include "xnnpack/XpInterface.hpp"
namespace mllm::xnnpack {

class XpBroadcastAddFunction : public TensorFunction, public XpTensorDefineInterface<XpBroadcastAddFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpBroadcastSubFunction : public TensorFunction, public XpTensorDefineInterface<XpBroadcastSubFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpBroadcastMulFunction : public TensorFunction, public XpTensorDefineInterface<XpBroadcastMulFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpBroadcastDivFunction : public TensorFunction, public XpTensorDefineInterface<XpBroadcastDivFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpTTAddFunction : public TensorFunction, public XpTensorDefineInterface<XpTTAddFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpTTSubFunction : public TensorFunction, public XpTensorDefineInterface<XpTTSubFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpTTMulFunction : public TensorFunction, public XpTensorDefineInterface<XpTTMulFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

class XpTTDivFunction : public TensorFunction, public XpTensorDefineInterface<XpTTDivFunction> {
public:
    void reshape(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;

    void execute(vector<shared_ptr<Tensor>> outputs, vector<shared_ptr<Tensor>> inputs, vector<float> args) override;
};

} // namespace mllm::xnnpack
