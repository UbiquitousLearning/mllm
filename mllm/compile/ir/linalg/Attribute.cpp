// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/compile/ir/linalg/Attribute.hpp"

namespace mllm::ir::linalg {

LinalgIRAttr::~LinalgIRAttr() = default;

LinalgIRAttr::LinalgIRAttr() : Attr(RK_Attr_LinalgIRAttr) {}

LinalgIRAttr::LinalgIRAttr(const NodeKind& kind) : Attr(kind) {}

LinalgIRQuantizatonAnnotationAttr::~LinalgIRQuantizatonAnnotationAttr() = default;

LinalgIRQuantizatonAnnotationAttr::LinalgIRQuantizatonAnnotationAttr()
    : LinalgIRAttr(RK_Attr_LinalgIRAttr_QuantizationAnnotation) {}

LinalgIRQuantizatonAnnotationAttr::LinalgIRQuantizatonAnnotationAttr(const NodeKind& kind) : LinalgIRAttr(kind) {}

}  // namespace mllm::ir::linalg
