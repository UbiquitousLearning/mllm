// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <cstdint>
#include <vector>
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/WeakOwner.hpp"

namespace mllm::nn {

enum class AbstractNnNodeTypes : int32_t {
  kLayer = 0,
  kModule = 1,
};

class AbstractNnNode : public std::enable_shared_from_this<AbstractNnNode> {
 public:
  using ptr_t = std::shared_ptr<AbstractNnNode>;

  virtual ~AbstractNnNode() = default;

  explicit AbstractNnNode(AbstractNnNodeTypes type);

  void regChildNode(const AbstractNnNode::ptr_t& child);

  WeakOwner<AbstractNnNode>& refParentNode();

  std::vector<AbstractNnNode::ptr_t>& refChildNodes();

  void setName(const std::string& name);

  void setAbsoluteName(const std::string& absolute_name);

  void setDepth(int32_t depth);

  void depthIncrease();

  void depthDecrease();

  [[nodiscard]] std::string getName() const;

  [[nodiscard]] std::string getAbsoluteName() const;

  [[nodiscard]] int32_t getDepth() const;

  [[nodiscard]] AbstractNnNodeTypes getType() const;

  [[nodiscard]] DeviceTypes getDevice() const;

  void setCompiledAsObj(bool flag);

  [[nodiscard]] bool isCompiledAsObj() const;

 protected:
  int32_t depth_ = 0;
  AbstractNnNodeTypes type_;
  bool compiled_as_obj_ = false;
  DeviceTypes device_type_ = kCPU;

  std::string name_;
  std::string absolute_name_;

  WeakOwner<AbstractNnNode> parent_node_ = nullptr;
  std::vector<AbstractNnNode::ptr_t> reg_child_nodes_;
};

}  // namespace mllm::nn