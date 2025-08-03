// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <memory>
#include <thread>

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
  SymbolTable<std::string, SessionContext::ptr_t> attached_contexts;
};

}  // namespace mllm
