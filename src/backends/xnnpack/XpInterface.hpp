/**
 * @file XpInterface.hpp
 * @author Chenghua Wang (chenghua.wang.edu@gmail.com)
 * @version 0.1
 * @date 2024-10-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once

#include "Tensor.hpp"
#include "xnnpack.h"
#include "backends/xnnpack/XnnpackBackend.hpp"
#include "backends/xnnpack/Utils/Logger.hpp"
#include <cassert>
#include <memory>

namespace mllm::xnnpack {

enum class XpTensorType : uint32_t {
    Normal = 0,
    ExternalInput = 1,
    ExternalOutput = 2,
};

template <typename T>
struct XpTensorDefineInterface {
    uint32_t defineTemporaryTensor(XnnpackCargo *xpb, const std::vector<size_t> &dims, DataType dtype) {
        auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(dtype);
        xnn_status status;

        uint32_t ret = XNN_INVALID_VALUE_ID;
        switch (xp_dtype) {
        case xnn_datatype_fp32: {
            status = xnn_define_tensor_value(
                xpb->getXnnSubgraph(), xp_dtype,
                dims.size(), dims.data(),
                nullptr,
                XNN_INVALID_VALUE_ID, 0, &ret);
            break;
        }
        default:
            break;
        }

        if (status != xnn_status_success) {
            Log::error("xnnpack backend defineTemporaryTensor Error");
            exit(-1);
        }

        return ret;
    }

    /**
     * @brief Define a Weight Tensor as External Input Tensor in order to avoid extra alloc each epoch in xnnpack
     *
     * @param xpb
     * @param t
     * @param forceDims
     */
    void defineWeightTensor(XnnpackCargo *xpb, Tensor *t, const std::vector<size_t> &forceDims = {}) {
        if (t->uuid() != XNN_INVALID_VALUE_ID) {
            if (xpb->hasExternalValue(t->uuid())) return;
        }

        auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

        xnn_status status;
        std::vector<size_t> dims;
        for (auto d : t->shape()) dims.push_back(d);

        if (!forceDims.empty()) {
            dims = forceDims;
        }

        switch (xp_dtype) {
        case xnn_datatype_fp32: {
            status = xnn_define_tensor_value(
                xpb->getXnnSubgraph(), xp_dtype,
                dims.size(), dims.data(),
                nullptr,
                xpb->getNewEXternalId(), XNN_VALUE_FLAG_EXTERNAL_INPUT, &t->uuid());
            break;
        }
        default:
            break;
        }

        if (status != xnn_status_success) {
            Log::error("xnnpack backend defineWeightTensor Error");
            exit(-1);
        }

        xpb->registerExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
        xpb->registerUuidWeightTensor(t->uuid(), t);
    }

    void tryDefineAllXpTensors(XnnpackCargo *xpb, const std::vector<std::shared_ptr<Tensor>> &ts) {
        for (auto &t : ts) {
            XpTensorType _t;
            switch (t->xnnTensorType()) {
            case TensorType::INPUT_TENSOR:
                _t = XpTensorType::ExternalInput;
                break;
            case TensorType::OUTPUT_TENSOR:
                _t = XpTensorType::ExternalOutput;
                break;
            case TensorType::NORMAL_TENSOR:
                _t = XpTensorType::Normal;
                break;
            }

            defineXpTensor(xpb, t.get(), _t);
        }
    }

    void tryDefineAllXpTensors(XnnpackCargo *xpb, const std::vector<Tensor *> &ts) {
        for (auto &t : ts) {
            XpTensorType _t;
            switch (t->xnnTensorType()) {
            case TensorType::INPUT_TENSOR:
                _t = XpTensorType::ExternalInput;
                break;
            case TensorType::OUTPUT_TENSOR:
                _t = XpTensorType::ExternalOutput;
                break;
            case TensorType::NORMAL_TENSOR:
                _t = XpTensorType::Normal;
                break;
            }

            defineXpTensor(xpb, t, _t);
        }
    }

    void defineXpTensor(XnnpackCargo *xpb, Tensor *t, XpTensorType ttype) {
        // for inputs and outputs
        if (xpb->inActivationName(t->name())) {
            t->uuid() = xpb->getUUIDByActivationName(t->name());
            xpb->updateExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            return;
        }

        if (xpb->getExecCnt()) return;

        // for normal values and weights.
        if (t->uuid() != XNN_INVALID_VALUE_ID) {
            if (xpb->hasExternalValue(t->uuid()) || xpb->hasNormalValue(t->uuid()) || xpb->hasWeightValue(t->uuid())) return;
        }

        auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

        xnn_status status;
        std::vector<size_t> dims;
        for (auto d : t->shape()) dims.push_back(d);

        uint32_t flags = 0;
        uint32_t external_id = XNN_INVALID_VALUE_ID;

        switch (ttype) {
        case XpTensorType::Normal:
            flags = 0;
            break;
        case XpTensorType::ExternalInput:
            flags = XNN_VALUE_FLAG_EXTERNAL_INPUT;
            external_id = xpb->getNewEXternalId();
            break;
        case XpTensorType::ExternalOutput:
            flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
            external_id = xpb->getNewEXternalId();
            break;
        }

        switch (xp_dtype) {
        case xnn_datatype_fp32: {
            status = xnn_define_tensor_value(
                xpb->getXnnSubgraph(), xp_dtype,
                dims.size(), dims.data(),
                /*data=*/nullptr,
                external_id, flags, &t->uuid());
            break;
        }
        default:
            break;
        }

        switch (ttype) {
        case XpTensorType::Normal:
            xpb->registerNormalValue(t->uuid());
            break;
        case XpTensorType::ExternalInput:
            xpb->registerExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            xpb->registerUuidTensor(t->uuid(), t);
            break;
        case XpTensorType::ExternalOutput:
            xpb->registerExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = nullptr});
            xpb->registerUuidTensor(t->uuid(), t);
            break;
        }

        if (status != xnn_status_success) {
            Log::error("xnnpack backend defineXpTensor Error");
            exit(-1);
        }
    }

    uint32_t defineKVCacheTensorAsExternalInput(XnnpackCargo *xpb, Tensor *t, int offset, const std::vector<size_t> &dims) {
        auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

        uint32_t flags = XNN_VALUE_FLAG_EXTERNAL_INPUT;
        uint32_t external_id = xpb->getNewEXternalId();

        uint32_t uuid;
        xnn_status status;

        switch (xp_dtype) {
        case xnn_datatype_fp32: {
            status = xnn_define_tensor_value(
                xpb->getXnnSubgraph(), xp_dtype,
                dims.size(), dims.data(),
                /*data=*/nullptr,
                external_id, flags, &uuid);
            xpb->registerExternalValue(uuid, xnn_external_value{.id = uuid, .data = t->hostPtr<float>() + offset});
            break;
        }
        default:
            break;
        }

        if (status != xnn_status_success) {
            Log::error("xnnpack backend defineKVCacheTensorAsExternalInput Error");
            exit(-1);
        }

        xpb->registerUuidTensor(uuid, t);

        return uuid;
    }

    uint32_t
    defineKVCacheTensorAsExternalOutput(XnnpackCargo *xpb, Tensor *t, int offset, const std::vector<size_t> &dims) {
        auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

        uint32_t flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
        uint32_t external_id = xpb->getNewEXternalId();

        uint32_t uuid;
        xnn_status status;

        switch (xp_dtype) {
        case xnn_datatype_fp32: {
            status = xnn_define_tensor_value(
                xpb->getXnnSubgraph(), xp_dtype,
                dims.size(), dims.data(),
                /*data=*/nullptr,
                external_id, flags, &uuid);
            xpb->registerExternalValue(uuid, xnn_external_value{.id = uuid, .data = t->hostPtr<float>() + offset});
            break;
        }
        default:
            break;
        }

        if (status != xnn_status_success) {
            Log::error("xnnpack backend defineKVCacheTensorAsExternalOutput Error");
            exit(-1);
        }

        xpb->registerUuidTensor(uuid, t);

        return uuid;
    }
};

} // namespace mllm::xnnpack