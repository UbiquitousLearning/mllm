// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"

namespace mllm::ir {

BuiltinIRAttr::~BuiltinIRAttr() = default;

BuiltinIRAttr::BuiltinIRAttr() : Attr(RK_Attr_BuiltinIRAttr) {}

BuiltinIRAttr::BuiltinIRAttr(const NodeKind& kind) : Attr(kind) {}

IntAttr::~IntAttr() = default;

IntAttr::IntAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_IntAttr) {}

IntAttr::IntAttr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

int& IntAttr::data() { return data_; }

IntAttr::ptr_t IntAttr::build(IRContext*, int data) {
  auto ret = std::make_shared<IntAttr>();
  ret->data() = data;
  return ret;
}

void IntAttr::dump(IRPrinter& p) { p.print("{}", data()); }

FPAttr::~FPAttr() = default;

FPAttr::FPAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_FPAttr) {}

FPAttr::FPAttr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

float& FPAttr::data() { return data_; }

FPAttr::ptr_t FPAttr::build(IRContext*, float data) {
  auto ret = std::make_shared<FPAttr>();
  ret->data() = data;
  return ret;
}

StrAttr::~StrAttr() = default;

StrAttr::StrAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_StrAttr) {}

StrAttr::StrAttr(const NodeKind& kind) : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_StrAttr) {}

std::string& StrAttr::data() { return data_; }

StrAttr::ptr_t StrAttr::build(IRContext*, const std::string& data) {
  auto ret = std::make_shared<StrAttr>();
  ret->data() = data;
  return ret;
}

void StrAttr::dump(IRPrinter& p) { p.print("{}", data()); }

SymbolAttr::~SymbolAttr() = default;

SymbolAttr::SymbolAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_SymbolAttr) {}

std::string& SymbolAttr::str() { return data_; }

void SymbolAttr::dump(IRPrinter& p) { p.print(data_); }

SymbolAttr::ptr_t SymbolAttr::build(IRContext*, const std::string& symbol_name) {
  auto ret = std::make_shared<SymbolAttr>();
  ret->str() = symbol_name;
  return ret;
}

BoolAttr::~BoolAttr() = default;

BoolAttr::BoolAttr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_BoolAttr) {}

BoolAttr::BoolAttr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

void BoolAttr::dump(IRPrinter& p) { p.print("{}", data()); }

bool& BoolAttr::data() { return data_; }

BoolAttr::ptr_t BoolAttr::build(IRContext*, bool data) {
  auto ret = std::make_shared<BoolAttr>();
  ret->data() = data;
  return ret;
}

VectorFP32Attr::VectorFP32Attr() : BuiltinIRAttr(RK_Attr_BuiltinIRAttr_VectorFP32Attr) {}

VectorFP32Attr::VectorFP32Attr(const NodeKind& kind) : BuiltinIRAttr(kind) {}

void VectorFP32Attr::dump(IRPrinter& p) { p.print("{}", data()); }

std::vector<float>& VectorFP32Attr::data() { return data_; }

VectorFP32Attr::ptr_t VectorFP32Attr::build(IRContext*, const std::vector<float>& data) {
  auto ret = std::make_shared<VectorFP32Attr>();
  ret->data() = data;
  return ret;
}

}  // namespace mllm::ir
