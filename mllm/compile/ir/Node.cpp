// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>

#include "mllm/compile/ir/Node.hpp"
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/IRPrinter.hpp"
#include "mllm/utils/RTTIHelper.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::ir {

Node::Node(const NodeKind& kind) : kind_(kind) {}

void Node::dumpAttributes(IRPrinter& p) {
  if (attrs_.empty()) return;
  p.lsbracket();
  auto size = attrs_.size();
  int cnt = 0;
  for (auto& [key, attr] : attrs_) {
    p.print(key);
    p.colon();
    attr->dump(p);
    if (cnt < size - 1) { p.comma(); }
    cnt++;
  }
  p.rsbracket();
}

std::list<node_weak_ptr_t>& Node::inputs() { return inputs_; }

std::list<node_weak_ptr_t>& Node::outputs() { return outputs_; }

node_weak_ptr_t Node::prevOp() { return prev_op_node_; }

void Node::setPrevOp(const node_weak_ptr_t& node) { prev_op_node_ = node; }

node_weak_ptr_t Node::nextOp() { return next_op_node_; }

void Node::setNextOp(const node_weak_ptr_t& node) { next_op_node_ = node; }

node_weak_ptr_t Node::belongsTo() { return belongs_to_parent_; }

void Node::setBelongsTo(const node_ptr_t& node) { belongs_to_parent_ = node; }

void Node::setBelongsTo(const node_weak_ptr_t& node) { belongs_to_parent_ = node; }

void Node::setAttr(const std::string& str, const attr_ptr_t& attr) { attrs_.insert({str, attr}); }

attr_ptr_t Node::getAttr(const std::string& str) {
  auto it = attrs_.find(str);
  return it != attrs_.end() ? it->second : nullptr;
}

Region::Region(const op_ptr_t& belongs_to) : belongs_to_(belongs_to) {}

std::list<op_ptr_t>& Region::ops() { return ops_; }

std::list<val_ptr_t>& Region::inputs() { return inputs_; }

std::list<val_ptr_t>& Region::outputs() { return outputs_; }

node_weak_ptr_t& Region::belongsTo() { return belongs_to_; }

void Region::dump(IRPrinter& p) {
  // inputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = inputs_.size();
    for (auto& ins : inputs_) {
      ins->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }

  // to
  IRPrinter::to();

  // outputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = outputs_.size();
    for (auto& ous : outputs_) {
      ous->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }

  p.blank();
  p.lbrace();

  // ops
  {
    size_t cnt = 0;
    auto size = ops_.size();
    for (auto& op : ops_) {
      op->dump(p);
      if (cnt < size - 1) p.newline();
      cnt++;
    }
  }

  p.rbrace();
}

Op::~Op() = default;

Op::Op() : Node(RK_Op) {};

Op::Op(const NodeKind& kind) : Node(kind) {};

void Op::dump(IRPrinter& p) {
  // inputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = inputs().size();
    for (auto& ins : inputs()) {
      ins->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }

  // to
  IRPrinter::to();

  // outputs
  {
    IRPrinter::lparentheses();
    size_t cnt = 0;
    auto size = outputs().size();
    for (auto& ous : outputs()) {
      ous->dump(p);
      if (cnt < size - 1) IRPrinter::comma();
      cnt++;
    }
    IRPrinter::rparentheses();
  }
}

std::shared_ptr<Region> Op::createRegionAtTop() {
  auto reg = std::make_shared<Region>(cast<Op>(shared_from_this()));
  regions_.push_back(reg);
  return reg;
}

std::list<std::shared_ptr<Region>>& Op::regions() { return regions_; }

std::shared_ptr<Region> Op::getTopRegion() { return regions_.back(); }

void Op::replacePartialOutputs(const std::vector<val_weak_ptr_t>& old_vals, const std::vector<val_weak_ptr_t>& new_vals) {
  MLLM_RT_ASSERT_EQ(old_vals.size(), new_vals.size());
  int val_num = old_vals.size();
  for (int i = 0; i < val_num; ++i) {
    auto old_val = old_vals[i];
    auto new_val = new_vals[i];
    // cut edge bt old_val and this op
    old_val->inputs().remove(shared_from_this());

    // Insert new_val to this op's outputs old_val position
    auto it = std::find(outputs().begin(), outputs().end(), old_val);
    MLLM_RT_ASSERT(it != outputs().end());
    *it = new_val;

    // new edge
    new_val->inputs().emplace_back(shared_from_this());
  }
}

void Op::replacePartialOutputs(val_weak_ptr_t old_vals, val_weak_ptr_t new_vals) {
  replacePartialOutputs(std::vector<val_weak_ptr_t>{old_vals}, std::vector<val_weak_ptr_t>{new_vals});
}

Attr::~Attr() = default;

Attr::Attr() : Node(RK_Attr) {};

Attr::Attr(const NodeKind& kind) : Node(kind) {}

void Attr::dump(IRPrinter& p) { p.print("<InvalidAttribute NYI>"); }

Val::~Val() = default;

Val::Val() : Node(RK_Val) {};

Val::Val(const NodeKind& kind) : Node(kind) {}

void Val::dump(IRPrinter& p) { p.print("%{}:", name()); }

std::string& Val::name() { return name_; }

op_weak_ptr_t Val::producerOp() {
  MLLM_RT_ASSERT_EQ(inputs().size(), 1);
  auto input = inputs().front();
  MLLM_RT_ASSERT(input->isa_<Op>());
  return input->cast_<Op>();
}

std::vector<op_weak_ptr_t> Val::consumerOps() {
  std::vector<op_weak_ptr_t> consumer_ops;
  for (auto& output : outputs()) {
    MLLM_RT_ASSERT(output->isa_<Op>());
    consumer_ops.emplace_back(output->cast_<Op>());
  }
  return consumer_ops;
}

IRContext::IRContext(const node_ptr_t& module, const region_ptr_t& region)
    : top_level_op_(module), cur_insert_region_(region) {}

void IRContext::setValueName(const val_ptr_t& val, const std::string& name) { value_names_[val] = name; }

std::string IRContext::getAutoIndexedValueName() { return std::to_string(auto_indexed_value_name_cnt_++); }

std::string IRContext::getUniqueModuleName(const std::string& base_name) {
  auto& counter = module_name_counters_[base_name];
  if (counter == 0) {
    counter = 1;
    return base_name;
  } else {
    return base_name + "_" + std::to_string(counter++);
  }
}

void IRContext::resetRegion(const region_ptr_t& region) { cur_insert_region_ = region; }

const IRContext::region_ptr_t& IRContext::getCurInsertRegion() { return cur_insert_region_; }

node_ptr_t& IRContext::topLevelOp() { return top_level_op_; }

void IRContext::removeFromSymbolTable(const std::string& name) { symbol_table_.erase(name); }

void IRContext::addToSymbolTable(const node_ptr_t& node, const std::string& name) { symbol_table_[name] = node; }

node_ptr_t IRContext::lookupSymbolTable(const std::string& name) {
  return symbol_table_.count(name) ? symbol_table_[name] : nullptr;
}

void IRContext::setDevice(DeviceTypes device_type) { device_type_ = device_type; }

// FIXME: deprecated, context has no device
DeviceTypes IRContext::getDevice() { return device_type_; }

bool IRContext::isCacheInputOutputTensor(uint32_t uuid) {
  if (cached_inputs_outputs_.count(uuid)) { return true; }
  return false;
}

void IRContext::cacheInputOutputTensor(uint32_t uuid, const val_ptr_t& tensor_ir) { cached_inputs_outputs_[uuid] = tensor_ir; }

val_ptr_t IRContext::getCacheInputOutputTensor(uint32_t uuid) { return cached_inputs_outputs_[uuid]; }

std::unordered_map<uint32_t, val_ptr_t>& IRContext::getAllCachedInputOutputTensorIRs() { return cached_inputs_outputs_; }

void IRContext::pushRegion2InsertRegionStackAndSetRegion(const region_ptr_t& region) {
  insert_region_stack_.push(region);
  cur_insert_region_ = region;
}

void IRContext::popRegionFromInsertRegionStackAndSetRegion() {
  cur_insert_region_ = insert_region_stack_.top();
  insert_region_stack_.pop();
}

IRWriterGuard::IRWriterGuard(const std::shared_ptr<IRContext>& ctx, const std::shared_ptr<Region>& new_region)
    : ctx_(ctx.get()), new_region_(new_region) {
  old_region_ = ctx->getCurInsertRegion();
  ctx->resetRegion(new_region_);
}

IRWriterGuard::IRWriterGuard(IRContext* ctx, const std::shared_ptr<Region>& new_region) : ctx_(ctx), new_region_(new_region) {
  old_region_ = ctx->getCurInsertRegion();
  ctx->resetRegion(new_region_);
}

IRWriterGuard::~IRWriterGuard() { ctx_->resetRegion(old_region_); }

IRWriter::IRWriter(const std::shared_ptr<IRContext>& ctx, const std::shared_ptr<Region>& cur_region)
    : ctx_(ctx), cur_region_(cur_region) {}

void IRWriter::removeValue(const val_ptr_t& val) {
  for (auto& input : val->inputs()) { input->outputs().remove(val); }
  for (auto& output : val->outputs()) { output->inputs().remove(val); }
}

void IRWriter::removeOp(const op_ptr_t& op) {
  auto& ops = cur_region_->ops();

  // Cutoff the edge
  for (auto& input : op->inputs()) { input->outputs().remove(op); }
  for (auto& output : op->outputs()) { output->inputs().remove(op); }

  auto it = std::find(ops.begin(), ops.end(), op);
  MLLM_RT_ASSERT(it != ops.end());
  cur_op_iter_ = ops.erase(it);

  is_iterator_modified_ = true;
}

void IRWriter::removeOpWithoutEdgeCut(const op_ptr_t& op) {
  auto& ops = cur_region_->ops();
  auto it = std::find(ops.begin(), ops.end(), op);
  MLLM_RT_ASSERT(it != ops.end());
  cur_op_iter_ = ops.erase(it);
  is_iterator_modified_ = true;
}

void IRWriter::replaceOp(const op_ptr_t& old_op, const op_ptr_t& new_op) {
  auto& ops = cur_region_->ops();

  for (auto& input : old_op->inputs()) { input->outputs().remove(old_op); }
  for (auto& output : old_op->outputs()) { output->inputs().remove(old_op); }

  for (auto& input : new_op->inputs()) { input->outputs().emplace_back(new_op); }
  for (auto& output : new_op->outputs()) { output->inputs().emplace_back(new_op); }

  new_op->setPrevOp(old_op->prevOp());
  new_op->setNextOp(old_op->nextOp());

  if (auto prev_op = old_op->prevOp()) { prev_op->setNextOp(new_op); }
  if (auto next_op = old_op->nextOp()) { next_op->setPrevOp(new_op); }

  new_op->setBelongsTo(old_op->belongsTo());
  std::replace(ops.begin(), ops.end(), old_op, new_op->template cast_<Op>());
}

void IRWriter::insertOpAtPos(const op_ptr_t& pos_op, Position pos, const op_ptr_t& new_op) {
  // find the pos_op iter in region
  auto& ops = cur_region_->ops();
  auto pos_op_iter = std::find(ops.begin(), ops.end(), pos_op);
  MLLM_RT_ASSERT(pos_op_iter != ops.end());
  new_op->setBelongsTo(pos_op->belongsTo());

  switch (pos) {
    case AFTER: {
      auto pre_op = *pos_op_iter;
      pos_op_iter++;
      auto next_op = (pos_op_iter != ops.end()) ? *pos_op_iter : nullptr;

      pre_op->setNextOp(new_op);
      new_op->setPrevOp(pre_op);
      if (next_op) next_op->setPrevOp(new_op);

      cur_op_iter_ = ops.insert(pos_op_iter, new_op);
      break;
    }
    case BEFORE: {
      auto next_op = *pos_op_iter;
      op_ptr_t pre_op = nullptr;
      if (pos_op_iter != ops.begin()) {
        pos_op_iter--;
        pre_op = *pos_op_iter;
      }

      if (pre_op) pre_op->setNextOp(new_op);
      new_op->setPrevOp(pre_op);
      new_op->setNextOp(next_op);
      if (next_op) next_op->setPrevOp(new_op);

      pos_op_iter = std::find(ops.begin(), ops.end(), next_op);
      cur_op_iter_ = ops.insert(pos_op_iter, new_op);
      break;
    }
  }
}

void IRWriter::insertOpAtLast(const op_ptr_t& new_op) {
  MLLM_RT_ASSERT(new_op != nullptr);
  auto& ops = cur_region_->ops();
  new_op->setBelongsTo(cur_region_->belongsTo());

  op_ptr_t last_op = ops.empty() ? nullptr : ops.back();
  if (last_op) {
    last_op->setNextOp(new_op);
    new_op->setPrevOp(last_op);
  } else {
    new_op->setPrevOp(nullptr);
  }
  new_op->setNextOp(nullptr);
  ops.push_back(new_op);
  cur_op_iter_ = std::prev(ops.end());
  is_iterator_modified_ = true;
}

void IRWriter::insertOpAtFront(const op_ptr_t& new_op) {
  MLLM_RT_ASSERT(new_op != nullptr);
  auto& ops = cur_region_->ops();
  new_op->setBelongsTo(cur_region_->belongsTo());
  op_ptr_t first_op = ops.empty() ? nullptr : ops.front();
  if (first_op) {
    first_op->setPrevOp(new_op);
    new_op->setNextOp(first_op);
  } else {
    new_op->setNextOp(nullptr);
  }
  new_op->setPrevOp(nullptr);
  ops.push_front(new_op);
  cur_op_iter_ = ops.begin();
  is_iterator_modified_ = true;
}

IRContext::ptr_t IRWriter::getContext() { return ctx_; }

}  // namespace mllm::ir
