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
#include "xnnpack/Utils/Logger.hpp"
#include <memory>
#include <array>

namespace mllm::xnnpack {

enum class XpTensorType : uint32_t {
    Normal = 0,
    ExternalInput = 1,
    ExternalOutput = 2,
};

template <typename T>
struct XpTensorDefineInterface {
    void defineXpTensor(XnnpackBackend *xpb, std::shared_ptr<Tensor> &t, XpTensorType ttype) {
        if (t->uuid() != XNN_INVALID_VALUE_ID) return;

        auto xp_dtype = XnnpackBackend::mllmDType2XnnDType(t->dtype());

        xnn_status status;
        std::array<size_t, 4> dims = {0, 0, 0, 0};
        dims[0] = t->batch();
        dims[1] = t->head();
        dims[2] = t->sequence();
        dims[3] = t->dimension();

        uint32_t flags;

        switch (ttype) {
        case XpTensorType::Normal:
            flags = XNN_INVALID_VALUE_ID;
            break;
        case XpTensorType::ExternalInput:
            flags = XNN_VALUE_FLAG_EXTERNAL_INPUT;
            break;
        case XpTensorType::ExternalOutput:
            flags = XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
            break;
        }

        switch (xp_dtype) {
        case xnn_datatype_fp32: {
            status = xnn_define_tensor_value(
                xpb->getXnnSubgraph(), xp_dtype,
                dims.size(), dims.data(),
                /*data=*/nullptr,
                0, flags, &t->uuid());
        }
        default:
            break;
        }

        switch (ttype) {
        case XpTensorType::Normal:
            break;
        case XpTensorType::ExternalInput:
        case XpTensorType::ExternalOutput:
            xpb->registerExternalValue(t->uuid(), xnn_external_value{.id = t->uuid(), .data = t->rawHostPtr()});
            break;
        }

        if (status != xnn_status_success) {
            Log::error("xnnpack backend defineXpTensor Error");
            exit(-1);
        }
    }
};

} // namespace mllm::xnnpack