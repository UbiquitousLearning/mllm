// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/nn/AbstractNnNode.hpp"
#include "mllm/utils/Dbg.hpp"

namespace mllm::nn {
AbstractNnNode::AbstractNnNode(AbstractNnNodeTypes type) : type_(type) {}

void AbstractNnNode::regChildNode(const AbstractNnNode::ptr_t& child) {
  // Link child node to parent node
  child->refParentNode() = shared_from_this();
  child->depthIncrease();
  reg_child_nodes_.emplace_back(child);
}

WeakOwner<AbstractNnNode>& AbstractNnNode::refParentNode() { return parent_node_; }

std::vector<AbstractNnNode::ptr_t>& AbstractNnNode::refChildNodes() { return reg_child_nodes_; }

void AbstractNnNode::setName(const std::string& name) { name_ = name; }

void AbstractNnNode::setAbsoluteName(const std::string& absolute_name) { absolute_name_ = absolute_name; }

void AbstractNnNode::setDepth(int32_t depth) { depth_ = depth; }

void AbstractNnNode::depthIncrease() {
  depth_++;
  for (auto& c : reg_child_nodes_) { c->depthIncrease(); }
}

void AbstractNnNode::depthDecrease() {
  depth_--;
  for (auto& c : reg_child_nodes_) { c->depthDecrease(); }
}

std::string AbstractNnNode::getName() const { return name_; }

std::string AbstractNnNode::getAbsoluteName() const { return absolute_name_; }

int32_t AbstractNnNode::getDepth() const { return depth_; }

AbstractNnNodeTypes AbstractNnNode::getType() const { return type_; }

DeviceTypes AbstractNnNode::getDevice() const { return device_type_; }

void AbstractNnNode::setCompiledAsObj(bool flag) { compiled_as_obj_ = flag; }

bool AbstractNnNode::isCompiledAsObj() const { return compiled_as_obj_; }
}  // namespace mllm::nn
