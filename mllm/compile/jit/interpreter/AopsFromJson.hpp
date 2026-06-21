// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <nlohmann/json_fwd.hpp>
#include "mllm/core/BaseOp.hpp"

namespace mllm::jit::interpreter {

BaseOp::ptr_t aopsFromJson(const nlohmann::json& json);

BaseOp::ptr_t __conv1dFromJson(const nlohmann::json& json);
BaseOp::ptr_t __conv3dFromJson(const nlohmann::json& json);
BaseOp::ptr_t __linearFromJson(const nlohmann::json& json);
BaseOp::ptr_t __matmulFromJson(const nlohmann::json& json);
BaseOp::ptr_t __fillFromJson(const nlohmann::json& json);
BaseOp::ptr_t __addFromJson(const nlohmann::json& json);
BaseOp::ptr_t __subFromJson(const nlohmann::json& json);
BaseOp::ptr_t __mulFromJson(const nlohmann::json& json);
BaseOp::ptr_t __divFromJson(const nlohmann::json& json);
BaseOp::ptr_t __absFromJson(const nlohmann::json& json);
BaseOp::ptr_t __logFromJson(const nlohmann::json& json);
BaseOp::ptr_t __embeddingFromJson(const nlohmann::json& json);
BaseOp::ptr_t __ropeFromJson(const nlohmann::json& json);
BaseOp::ptr_t __kvCacheFromJson(const nlohmann::json& json);
BaseOp::ptr_t __causalMaskFromJson(const nlohmann::json& json);
BaseOp::ptr_t __softmaxFromJson(const nlohmann::json& json);
BaseOp::ptr_t __transposeFromJson(const nlohmann::json& json);
BaseOp::ptr_t __rmsNormFromJson(const nlohmann::json& json);
BaseOp::ptr_t __siluFromJson(const nlohmann::json& json);
BaseOp::ptr_t __castTypeFromJson(const nlohmann::json& json);
BaseOp::ptr_t __x2xFromJson(const nlohmann::json& json);
BaseOp::ptr_t __viewFromJson(const nlohmann::json& json);
BaseOp::ptr_t __splitFromJson(const nlohmann::json& json);
BaseOp::ptr_t __stftFromJson(const nlohmann::json& json);
BaseOp::ptr_t __flashAttention2FromJson(const nlohmann::json& json);
BaseOp::ptr_t __repeatFromJson(const nlohmann::json& json);
BaseOp::ptr_t __permuteFromJson(const nlohmann::json& json);
BaseOp::ptr_t __conv2dFromJson(const nlohmann::json& json);
BaseOp::ptr_t __geluFromJson(const nlohmann::json& json);
BaseOp::ptr_t __layerNormFromJson(const nlohmann::json& json);
BaseOp::ptr_t __multimodalRopeFromJson(const nlohmann::json& json);
BaseOp::ptr_t __visionRopeFromJson(const nlohmann::json& json);
BaseOp::ptr_t __quickGeluFromJson(const nlohmann::json& json);
BaseOp::ptr_t __copyFromJson(const nlohmann::json& json);
BaseOp::ptr_t __cloneFromJson(const nlohmann::json& json);
BaseOp::ptr_t __negFromJson(const nlohmann::json& json);
BaseOp::ptr_t __concatFromJson(const nlohmann::json& json);
BaseOp::ptr_t __reduceMaxFromJson(const nlohmann::json& json);
BaseOp::ptr_t __reduceMinFromJson(const nlohmann::json& json);
BaseOp::ptr_t __reduceSumFromJson(const nlohmann::json& json);
BaseOp::ptr_t __reluFromJson(const nlohmann::json& json);
BaseOp::ptr_t __contiguousFromJson(const nlohmann::json& json);
BaseOp::ptr_t __reshapeFromJson(const nlohmann::json& json);
BaseOp::ptr_t __sliceFromJson(const nlohmann::json& json);
BaseOp::ptr_t __paramFromJson(const nlohmann::json& json);
BaseOp::ptr_t __indexFromJson(const nlohmann::json& json);
BaseOp::ptr_t __topkFromJson(const nlohmann::json& json);
BaseOp::ptr_t __meanFromJson(const nlohmann::json& json);
BaseOp::ptr_t __clipFromJson(const nlohmann::json& json);
BaseOp::ptr_t __expFromJson(const nlohmann::json& json);
BaseOp::ptr_t __sinFromJson(const nlohmann::json& json);
BaseOp::ptr_t __cosFromJson(const nlohmann::json& json);

}  // namespace mllm::jit::interpreter
