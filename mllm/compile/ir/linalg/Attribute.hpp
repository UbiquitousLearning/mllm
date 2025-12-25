// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <utility>

#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/NodeRTTIClassOfImpl.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/Tensor.hpp"

namespace mllm::ir::linalg {

class LinalgIRAttr : public Attr {
 public:
  DEFINE_SPECIFIC_IR_CLASS(LinalgIRAttr);

  ~LinalgIRAttr() override;

  LinalgIRAttr();

  explicit LinalgIRAttr(const NodeKind& kind);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_LINALGIRATTR_IMPL(node); }
};

enum class QuantizationSpecType : uint32_t {
  kNone = 0,
  kRaw,
  kSymPerTensor,
  kSymPerChannel,
  kSymPerBlock,
  kAsymPerTensor,
  kAsymPerChannel,
  kAsymPerBlock,
  kLPBQ,
};

class QuantizationSpecUUIDGiver {
 public:
  static QuantizationSpecUUIDGiver& getInstance() {
    static QuantizationSpecUUIDGiver instance;
    return instance;
  }

  uint64_t getUUID() { return next_uuid_++; }

 private:
  QuantizationSpecUUIDGiver() = default;
  QuantizationSpecUUIDGiver(const QuantizationSpecUUIDGiver&) = delete;             // NOLINT
  QuantizationSpecUUIDGiver& operator=(const QuantizationSpecUUIDGiver&) = delete;  // NOLINT

  uint64_t next_uuid_ = 0;
};

struct QuantizationSpec {
  using ptr_t = std::shared_ptr<QuantizationSpec>;
  QuantizationSpecType type;
  uint64_t uuid;
};

struct QuantizationSpecRaw : public QuantizationSpec {
  DataTypes type_ = kFloat32;

  static inline ptr_t create(DataTypes type) {
    auto spec = std::make_shared<QuantizationSpecRaw>();
    spec->type = QuantizationSpecType::kRaw;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    spec->type_ = type;
    return spec;
  }

  static inline ptr_t create() { return create(kFloat32); }
};

struct QuantizationSpecSymPerTensor : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_type = kFloat32;
  Tensor scale = Tensor::nil();

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, DataTypes quant_to_type, DataTypes scale_type,
                             Tensor scale) {
    auto spec = std::make_shared<QuantizationSpecSymPerTensor>();
    spec->type = QuantizationSpecType::kSymPerTensor;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->quant_to_type = quant_to_type;
    spec->scale_type = scale_type;
    spec->scale = std::move(scale);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecSymPerTensor>();
    spec->type = QuantizationSpecType::kSymPerTensor;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    return spec;
  }
};

struct QuantizationSpecSymPerChannel : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  int32_t ch_axis = -1;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_type = kFloat32;
  Tensor scale = Tensor::nil();

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, int32_t ch_axis, DataTypes quant_to_type,
                             DataTypes scale_type, Tensor scale) {
    auto spec = std::make_shared<QuantizationSpecSymPerChannel>();
    spec->type = QuantizationSpecType::kSymPerChannel;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->ch_axis = ch_axis;
    spec->quant_to_type = quant_to_type;
    spec->scale_type = scale_type;
    spec->scale = std::move(scale);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecSymPerChannel>();
    spec->type = QuantizationSpecType::kSymPerChannel;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    return spec;
  }
};

struct QuantizationSpecSymPerBlock : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  int32_t block_size = -1;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_type = kFloat32;
  Tensor scale = Tensor::nil();  ///< Flattened scale, blocks num

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, int32_t block_size, DataTypes quant_to_type,
                             DataTypes scale_type, Tensor scale) {
    auto spec = std::make_shared<QuantizationSpecSymPerBlock>();
    spec->type = QuantizationSpecType::kSymPerBlock;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->block_size = block_size;
    spec->quant_to_type = quant_to_type;
    spec->scale_type = scale_type;
    spec->scale = std::move(scale);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecSymPerBlock>();
    spec->type = QuantizationSpecType::kSymPerBlock;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    return spec;
  }
};

struct QuantizationSpecAsymPerTensor : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_type = kFloat32;
  DataTypes zero_point_type = kInt32;
  Tensor scale = Tensor::nil();
  Tensor zero_point = Tensor::nil();

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, DataTypes quant_to_type, DataTypes scale_type,
                             DataTypes zero_point_type, Tensor scale, Tensor zero_point) {
    auto spec = std::make_shared<QuantizationSpecAsymPerTensor>();
    spec->type = QuantizationSpecType::kAsymPerTensor;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->quant_to_type = quant_to_type;
    spec->scale_type = scale_type;
    spec->zero_point_type = zero_point_type;
    spec->scale = std::move(scale);
    spec->zero_point = std::move(zero_point);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecAsymPerTensor>();
    spec->type = QuantizationSpecType::kAsymPerTensor;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    return spec;
  }
};

struct QuantizationSpecAsymPerChannel : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  int32_t ch_axis = -1;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_type = kFloat32;
  DataTypes zero_point_type = kInt32;
  Tensor scale = Tensor::nil();
  Tensor zero_point = Tensor::nil();

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, int32_t ch_axis, DataTypes quant_to_type,
                             DataTypes scale_type, DataTypes zero_point_type, Tensor scale, Tensor zero_point) {
    auto spec = std::make_shared<QuantizationSpecAsymPerChannel>();
    spec->type = QuantizationSpecType::kAsymPerChannel;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->ch_axis = ch_axis;
    spec->quant_to_type = quant_to_type;
    spec->scale_type = scale_type;
    spec->zero_point_type = zero_point_type;
    spec->scale = std::move(scale);
    spec->zero_point = std::move(zero_point);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecAsymPerChannel>();
    spec->type = QuantizationSpecType::kAsymPerChannel;
    spec->uuid = QuantizationSpecUUIDGiver::getInstance().getUUID();
    return spec;
  }
};

struct QuantizationSpecAsymPerBlock : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  int32_t block_size = -1;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_type = kFloat32;
  DataTypes zero_point_type = kInt32;
  Tensor scale = Tensor::nil();       ///< Flattened scale, blocks num
  Tensor zero_point = Tensor::nil();  ///< Flattened zero_point, blocks num

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, int32_t block_size, DataTypes quant_to_type,
                             DataTypes scale_type, DataTypes zero_point_type, Tensor scale, Tensor zero_point) {
    auto spec = std::make_shared<QuantizationSpecAsymPerBlock>();
    spec->type = QuantizationSpecType::kAsymPerBlock;
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->block_size = block_size;
    spec->quant_to_type = quant_to_type;
    spec->scale_type = scale_type;
    spec->zero_point_type = zero_point_type;
    spec->scale = std::move(scale);
    spec->zero_point = std::move(zero_point);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecAsymPerBlock>();
    spec->type = QuantizationSpecType::kAsymPerBlock;
    return spec;
  }
};

struct QuantizationSpecLPBQ : public QuantizationSpec {
  int32_t quant_min = -1;
  int32_t quant_max = -1;
  int32_t block_size = -1;
  int32_t ch_axis = -1;
  int32_t scale_level_0_bitwidth = 4;
  DataTypes quant_to_type = kUInt8;
  DataTypes scale_1_type = kFloat32;
  Tensor scale_level_0_int = Tensor::nil();  ///< Flattened scale, blocks num
  Tensor scale_level_1_fp = Tensor::nil();   ///< Flattened scale, channel num

  static inline ptr_t create(int32_t quant_min, int32_t quant_max, int32_t block_size, int32_t ch_axis,
                             int32_t scale_level_0_bitwidth, DataTypes quant_to_type, DataTypes scale_1_type,
                             Tensor scale_level_0_int, Tensor scale_level_1_fp) {
    auto spec = std::make_shared<QuantizationSpecLPBQ>();
    spec->type = QuantizationSpecType::kLPBQ;
    spec->quant_min = quant_min;
    spec->quant_max = quant_max;
    spec->block_size = block_size;
    spec->ch_axis = ch_axis;
    spec->scale_level_0_bitwidth = scale_level_0_bitwidth;
    spec->quant_to_type = quant_to_type;
    spec->scale_1_type = scale_1_type;
    spec->scale_level_0_int = std::move(scale_level_0_int);
    spec->scale_level_1_fp = std::move(scale_level_1_fp);
    return spec;
  }

  static inline ptr_t create() {
    auto spec = std::make_shared<QuantizationSpecLPBQ>();
    spec->type = QuantizationSpecType::kLPBQ;
    return spec;
  }
};

struct QuantizationAnnotation {
  std::vector<QuantizationSpec::ptr_t> inputs;
  std::vector<QuantizationSpec::ptr_t> outputs;
  std::unordered_map<std::string, QuantizationSpec::ptr_t> weights;
};

class LinalgIRQuantizatonAnnotationAttr final : public LinalgIRAttr {
 public:
  QuantizationAnnotation annotation_;

  DEFINE_SPECIFIC_IR_CLASS(LinalgIRQuantizatonAnnotationAttr);

  ~LinalgIRQuantizatonAnnotationAttr() override;

  LinalgIRQuantizatonAnnotationAttr();

  explicit LinalgIRQuantizatonAnnotationAttr(const NodeKind& kind);

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_LINALGIRATTR_QUANTIZATIONANNOTATION_IMPL(node); }
};

class LinalgIRQuantizatonSpecAttr final : public LinalgIRAttr {
 public:
  QuantizationSpec::ptr_t spec_;

  DEFINE_SPECIFIC_IR_CLASS(LinalgIRQuantizatonSpecAttr);

  ~LinalgIRQuantizatonSpecAttr() override;

  LinalgIRQuantizatonSpecAttr();

  explicit LinalgIRQuantizatonSpecAttr(const NodeKind& kind);

  void dump(IRPrinter& p) override;

  static ptr_t build(IRContext* ctx);

  static ptr_t build(IRContext* ctx, const QuantizationSpec::ptr_t& spec);

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_LINALGIRATTR_QUANTIZATIONSPEC_IMPL(node); }
};

}  // namespace mllm::ir::linalg
