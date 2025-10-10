// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <stack>
#include <string>
#include <memory>
#include <algorithm>
#include <functional>
#include <unordered_map>

#include "mllm/core/DeviceTypes.hpp"
#include "mllm/utils/RTTIHelper.hpp"
#include "mllm/compile/ir/IRPrinter.hpp"

// include auto generated rtti kinds.
#include "mllm/compile/ir/GeneratedRTTIKind.hpp"
#include "mllm/compile/ir/NodeRTTIClassOfImpl.hpp"
#include "mllm/utils/WeakOwner.hpp"

#define DEFINE_SPECIFIC_IR_CLASS(_Type) using ptr_t = std::shared_ptr<_Type>

namespace mllm::ir {

class IRContext;
class Attr;
class Op;
class Val;
class Node;

using node_ptr_t = std::shared_ptr<Node>;

using node_weak_ptr_t = WeakOwner<Node>;

using op_ptr_t = std::shared_ptr<Op>;

using op_weak_ptr_t = WeakOwner<Op>;

using val_ptr_t = std::shared_ptr<Val>;

using val_weak_ptr_t = WeakOwner<Val>;

using attr_ptr_t = std::shared_ptr<Attr>;

using attr_weak_ptr_t = std::shared_ptr<Attr>;

class Node : public std::enable_shared_from_this<Node> {
 public:
  Node() = default;

  explicit Node(const NodeKind& kind);
  virtual ~Node() = default;

  virtual void dump(IRPrinter& p) { p.print("<InvalidNodePrinter NIY>"); };

  void dumpAttributes(IRPrinter& p);

  NodeKind getKind() const { return kind_; }

  // the code below impl the feature like this
  // node_ptr_t a, b;
  // (*a) --> b
  inline Node& operator--(int _) { return *shared_from_this(); }

  inline Node& operator>(const node_ptr_t& rhs) {
    this->outputs_.emplace_back(rhs);
    rhs->inputs_.emplace_back(shared_from_this());
    return *rhs;
  }

  std::list<node_weak_ptr_t>& inputs();

  std::list<node_weak_ptr_t>& outputs();

  node_weak_ptr_t prevOp();

  void setPrevOp(const node_weak_ptr_t& node);

  node_weak_ptr_t nextOp();

  void setNextOp(const node_weak_ptr_t& node);

  node_weak_ptr_t belongsTo();

  void setBelongsTo(const node_ptr_t& node);

  void setBelongsTo(const node_weak_ptr_t& node);

  void setAttr(const std::string& str, const attr_ptr_t& attr);

  attr_ptr_t getAttr(const std::string& str);

  template<typename T>
  bool isa_() {
    return isa<T>(shared_from_this());
  }

  template<typename T>
  std::shared_ptr<T> cast_() {
    return cast<T>(shared_from_this());
  }

 private:
  NodeKind kind_ = RK_None;
  node_weak_ptr_t prev_op_node_ = nullptr;
  node_weak_ptr_t next_op_node_ = nullptr;
  node_weak_ptr_t belongs_to_parent_ = nullptr;
  std::list<node_weak_ptr_t> inputs_;
  std::list<node_weak_ptr_t> outputs_;
  std::unordered_map<std::string, attr_ptr_t> attrs_;
};

class Region : public std::enable_shared_from_this<Region> {
 public:
  using ptr_t = std::shared_ptr<Region>;

  explicit Region(const op_ptr_t& belongs_to);

  std::list<op_ptr_t>& ops();

  std::list<val_ptr_t>& inputs();

  std::list<val_ptr_t>& outputs();

  node_weak_ptr_t& belongsTo();

  void dump(IRPrinter& p);

 private:
  node_weak_ptr_t belongs_to_ = nullptr;
  std::list<op_ptr_t> ops_;
  std::list<val_ptr_t> inputs_;
  std::list<val_ptr_t> outputs_;
};

template<typename T>
class DeviceInterface {
 public:
  void setDevice(DeviceTypes device_type) { device_type_ = device_type; }

  DeviceTypes getDevice() { return device_type_; }

  bool hasDevice() { return true; }

 private:
  DeviceTypes device_type_;
};

class Op : public Node, public DeviceInterface<Op> {
 public:
  using ptr_t = op_ptr_t;

  ~Op() override;

  Op();

  explicit Op(const NodeKind& kind);

  void dump(IRPrinter& p) override;

  static inline bool classof(const Node* node) { RTTI_RK_OP_IMPL(node); }

  std::shared_ptr<Region> createRegionAtTop();

  std::list<std::shared_ptr<Region>>& regions();

  std::shared_ptr<Region> getTopRegion();

  void replacePartialOutputs(const std::vector<val_weak_ptr_t>& old_vals, const std::vector<val_weak_ptr_t>& new_vals);

  void replacePartialOutputs(val_weak_ptr_t old_vals, val_weak_ptr_t new_vals);

 private:
  std::list<std::shared_ptr<Region>> regions_;
};

class Attr : public Node {
 public:
  ~Attr() override;
  Attr();
  explicit Attr(const NodeKind& kind);

  void dump(IRPrinter& p) override;

  static inline bool classof(const Node* node) { RTTI_RK_ATTR_IMPL(node); }
};

class Val : public Node, public DeviceInterface<Val> {
 public:
  ~Val() override;
  Val();
  explicit Val(const NodeKind& kind);

  void dump(IRPrinter& p) override;

  static inline bool classof(const Node* node) { RTTI_RK_VAL_IMPL(node); }

  std::string& name();

  op_weak_ptr_t producerOp();

  std::vector<op_weak_ptr_t> consumerOps();

 private:
  std::string name_;
};

class IRContext : public std::enable_shared_from_this<IRContext> {
  using region_ptr_t = std::shared_ptr<Region>;

 public:
  using ptr_t = std::shared_ptr<IRContext>;

  IRContext() = default;

  explicit IRContext(const node_ptr_t& module, const region_ptr_t& region);

 public:
  void setValueName(const val_ptr_t& val, const std::string& name);

  std::string getAutoIndexedValueName();

  void resetRegion(const region_ptr_t& region);

  const region_ptr_t& getCurInsertRegion();

  node_ptr_t& topLevelOp();

  template<typename T, typename... Args>
  std::shared_ptr<T> createAndSetModuleOp(Args&&... args) {
    auto _this = shared_from_this();
    std::shared_ptr<T> created_node = T::build(_this.get(), std::forward<Args>(args)...);
    topLevelOp() = created_node;

    addToSymbolTable(created_node, "main");

    cur_insert_region_ = created_node->template cast_<Op>()->getTopRegion();

    return created_node;
  }

  template<typename T, typename... Args>
  std::shared_ptr<T> createTemporaryValue(Args&&... args) {
    auto _this = shared_from_this();

    std::shared_ptr<T> created_node = T::build(_this.get(), std::forward<Args>(args)...);

    if (created_node->template cast_<Val>()->name().empty()) {
      setValueName(created_node->template cast_<Val>(), getAutoIndexedValueName());
      created_node->template cast_<Val>()->name() = value_names_[created_node->template cast_<Val>()];
    } else {
      setValueName(created_node->template cast_<Val>(), created_node->template cast_<Val>()->name());
    }

    return created_node;
  }

  template<typename T, typename... Args>
  std::shared_ptr<T> create(Args&&... args) {
    auto _this = shared_from_this();

    std::shared_ptr<T> created_node = T::build(_this.get(), std::forward<Args>(args)...);

    // Op: insert into region
    if (created_node->template isa_<Op>()) {
      auto& ops = cur_insert_region_->ops();
      op_ptr_t prev_op = ops.empty() ? nullptr : ops.back();

      cur_insert_region_->ops().push_back(created_node->template cast_<Op>());

      // belongsto
      created_node->setBelongsTo(cur_insert_region_->belongsTo());

      // set prev op
      created_node->setPrevOp(prev_op);

      // set next op
      if (prev_op) prev_op->setNextOp(created_node);
    }

    // Value: add to symbol table. Giving them names.
    if (created_node->template isa_<Val>()) {
      if (created_node->template cast_<Val>()->name().empty()) {
        setValueName(created_node->template cast_<Val>(), getAutoIndexedValueName());
        created_node->template cast_<Val>()->name() = value_names_[created_node->template cast_<Val>()];
      } else {
        setValueName(created_node->template cast_<Val>(), created_node->template cast_<Val>()->name());
      }
    }

    // Attribute: do nothing

    return created_node;
  }

  void removeFromSymbolTable(const std::string& name);

  void addToSymbolTable(const node_ptr_t& node, const std::string& name);

  node_ptr_t lookupSymbolTable(const std::string& name);

  void setDevice(DeviceTypes device_type);

  // FIXME: deprecated, context has no device
  DeviceTypes getDevice();

  bool isCacheInputOutputTensor(uint32_t uuid);

  void cacheInputOutputTensor(uint32_t uuid, const val_ptr_t& tensor_ir);

  val_ptr_t getCacheInputOutputTensor(uint32_t uuid);

  std::unordered_map<uint32_t, val_ptr_t>& getAllCachedInputOutputTensorIRs();

  // A long name that avoid user to use this low level API
  void pushRegion2InsertRegionStackAndSetRegion(const region_ptr_t& region);

  void popRegionFromInsertRegionStackAndSetRegion();

 private:
  DeviceTypes device_type_ = kCPU;  // FIXME: deprecated, context has no device
  std::unordered_map<std::string, node_ptr_t> symbol_table_;
  uint32_t auto_indexed_value_name_cnt_ = 0;
  std::unordered_map<val_ptr_t, std::string> value_names_;
  region_ptr_t cur_insert_region_;
  node_ptr_t top_level_op_;
  std::unordered_map<uint32_t, val_ptr_t> cached_inputs_outputs_;
  std::stack<region_ptr_t> insert_region_stack_;
};

class IRWriterGuard {
 public:
  IRWriterGuard(const std::shared_ptr<IRContext>& ctx, const std::shared_ptr<Region>& new_region);

  IRWriterGuard(IRContext* ctx, const std::shared_ptr<Region>& new_region);

  ~IRWriterGuard();

 private:
  IRContext* ctx_;
  std::shared_ptr<Region> old_region_ = nullptr;
  std::shared_ptr<Region> new_region_ = nullptr;
};

class IRWriter {
 public:
  IRWriter() = delete;
  IRWriter(const std::shared_ptr<IRContext>& ctx, const std::shared_ptr<Region>& cur_region);

  enum Position {
    AFTER = 0,
    BEFORE = 1,
  };

  enum WalkResult {
    WALK_CONTINUE = 0,
    WALK_BREAK = 1,
  };

  // create op at the end of a region
  template<typename T, typename... Args>
  std::shared_ptr<T> create(Args&&... args) {
    auto _this = ctx_.get();

    std::shared_ptr<T> created_node = T::build(_this, std::forward<Args>(args)...);

    // Op: insert into region
    if (created_node->template isa_<Op>()) {
      auto& ops = cur_region_->ops();
      op_ptr_t prev_op = ops.empty() ? nullptr : ops.back();

      cur_region_->ops().push_back(created_node->template cast_<Op>());

      // belongsto
      created_node->setBelongsTo(ctx_->getCurInsertRegion()->belongsTo());

      // set prev op
      created_node->setPrevOp(prev_op);

      // set next op
      if (prev_op) prev_op->setNextOp(created_node);
    }

    // Value: add to symbol table. Giving them names.
    if (created_node->template isa_<Val>()) {
      auto name = ctx_->getAutoIndexedValueName();
      ctx_->setValueName(created_node->template cast_<Val>(), name);
      created_node->template cast_<Val>()->name() = name;
    }

    // Attribute: do nothing

    return created_node;
  }

  // insert already exists op at
  void insertOpAtPos(const op_ptr_t& pos_op, Position pos, const op_ptr_t& new_op);

  // create op before an op
  template<typename T, typename... Args>
  std::shared_ptr<T> createAtPos(const op_ptr_t& pos_op, Position pos, Args&&... args) {
    auto _this = ctx_.get();

    std::shared_ptr<T> created_node = T::build(_this, std::forward<Args>(args)...);

    // find the pos_op iter in region
    auto& ops = cur_region_->ops();
    auto pos_op_iter = std::find(ops.begin(), ops.end(), pos_op);

    // Op: insert into region
    if (created_node->template isa_<Op>()) {
      switch (pos) {
        case AFTER: {
          auto pre_op = *pos_op_iter;
          pos_op_iter++;
          auto next_op = (pos_op_iter != ops.end()) ? *pos_op_iter : nullptr;

          pre_op->setNextOp(created_node);
          created_node->setPrevOp(pre_op);
          if (next_op) next_op->setPrevOp(created_node);

          cur_op_iter_ = ops.insert(pos_op_iter, created_node);
          break;
        }
        case BEFORE: {
          auto next_op = *pos_op_iter;
          op_ptr_t pre_op = nullptr;
          if (pos_op_iter != ops.begin()) {
            pos_op_iter--;
            pre_op = *pos_op_iter;
          }

          if (pre_op) pre_op->setNextOp(created_node);
          created_node->setPrevOp(pre_op);
          created_node->setNextOp(next_op);
          if (next_op) next_op->setPrevOp(created_node);

          pos_op_iter = std::find(ops.begin(), ops.end(), next_op);
          cur_op_iter_ = ops.insert(pos_op_iter, created_node);
          break;
        }
      }

      is_iterator_modified_ = true;
    }

    // Value: add to symbol table. Giving them names.
    if (created_node->template isa_<Val>()) {
      auto name = ctx_->getAutoIndexedValueName();
      ctx_->setValueName(created_node->template cast_<Val>(), name);
      created_node->template cast_<Val>()->name() = name;
    }

    // Attribute: do nothing

    return created_node;
  }

  void removeValue(const val_ptr_t& val);

  void removeOp(const op_ptr_t& op);

  void removeOpWithoutEdgeCut(const op_ptr_t& op);

  void replaceOp(const op_ptr_t& old_op, const op_ptr_t& new_op);

  void insertOpAtLast(const op_ptr_t& new_op);

  void insertOpAtFront(const op_ptr_t& new_op);

  template<typename T, typename... Args>
  std::shared_ptr<T> createAndReplaceOp(const op_ptr_t& old_op, Args&&... args) {
    auto _this = ctx_.get();

    std::shared_ptr<T> created_node = T::build(_this, std::forward<Args>(args)...);

    // Op: insert into region
    if (created_node->template isa_<Op>()) {
      // Cut edge
      for (auto& input : old_op->inputs()) { input->outputs().remove(old_op); }
      for (auto& output : old_op->outputs()) { output->inputs().remove(old_op); }

      auto prev_op = old_op->prevOp();
      auto next_op = old_op->nextOp();

      // belongsto
      created_node->setBelongsTo(ctx_->getCurInsertRegion()->belongsTo());

      // set prev op
      created_node->setPrevOp(prev_op);

      // set next op
      if (prev_op) prev_op->setNextOp(created_node);
      created_node->setNextOp(next_op);
      if (next_op) next_op->setPrevOp(created_node);
    }

    auto& ops = cur_region_->ops();
    std::replace(ops.begin(), ops.end(), old_op, created_node->template cast_<Op>());

    return created_node;
  }

  template<typename T>
  bool walk(const std::function<WalkResult(IRWriter&, const std::shared_ptr<T>&)>& func) {
    // get op
    auto& ops = cur_region_->ops();

    // loop
    for (cur_op_iter_ = ops.begin(); cur_op_iter_ != ops.end(); /*do nothing*/) {
      auto& op = *cur_op_iter_;
      if (op->isa_<T>()) {
        is_iterator_modified_ = false;

        auto ret = func(*this, op->cast_<T>());
        switch (ret) {
          case WALK_CONTINUE: break;
          case WALK_BREAK: return false;
        }

        if (!is_iterator_modified_) cur_op_iter_++;
      } else {
        cur_op_iter_++;
      }
    }

    return true;
  }

  IRContext::ptr_t getContext();

 private:
  bool is_iterator_modified_ = false;
  std::list<op_ptr_t>::iterator cur_op_iter_;
  Region::ptr_t cur_region_ = nullptr;
  IRContext::ptr_t ctx_ = nullptr;
};

}  // namespace mllm::ir
