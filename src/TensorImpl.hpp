/**
 * @file TensorImpl.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-11-11
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include <memory>

namespace mllm {

template <typename T>
struct TensorImplBase : public std::enable_shared_from_this<TensorImplBase<T>> {
    std::shared_ptr<T> self() {
        return std::static_pointer_cast<T>(this->shared_from_this());
    }

    template <typename... Args>
    static std::shared_ptr<T> create(Args... args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
};

class QuantizedPerTensorImpl : public TensorImplBase<QuantizedPerTensorImpl> {
public:
    explicit QuantizedPerTensorImpl(float scale = 0.f) :
        scale_(scale) {
    }

    void setScale(float scale) {
        scale_ = scale;
    }

    float getScale() const {
        return scale_;
    }

private:
    float scale_;
};

class QunatizedPerChannelImpl : public TensorImplBase<QunatizedPerChannelImpl> {
public:
    explicit QunatizedPerChannelImpl(float *scales = {}) :
        scales_(scales) {
    }

    void setScale(float *scales) {
        scales_ = scales;
    }

    float *getScale() const {
        return scales_;
    }

private:
    float *scales_;
};

} // namespace mllm