/**
 * @file SessionTCB.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-23
 *
 */
#pragma once

#include <string>
#include <memory>
#include <thread>

#include "mllm/core/BaseOp.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/compile/ir/Node.hpp"

namespace mllm {

class SessionContext : public std::enable_shared_from_this<SessionContext> {
 public:
  using ptr_t = std::shared_ptr<SessionContext>;
};

/**
 * @brief Session Thread Control Block
 *
 */
class SessionTCB {
 public:
  using ptr_t = std::shared_ptr<SessionTCB>;

  bool trace_mode = false;

  std::thread::id system_tid;
  ir::IRContext::ptr_t ir_context = nullptr;
  SymbolTable<std::string, BaseOp::ptr_t> layer_ops_table;
  SymbolTable<std::string, SessionContext::ptr_t> attached_contexts;
};

}  // namespace mllm
