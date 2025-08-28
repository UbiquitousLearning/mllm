// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory>

#include "mllm/compile/ir/Trace.hpp"
#include "mllm/compile/ir/builtin/Attribute.hpp"
#include "mllm/compile/ir/builtin/Op.hpp"
#include "mllm/compile/ir/IRTraceDispatcher.hpp"
#include "mllm/compile/ir/dbg/Op.hpp"
#include "mllm/compile/ir/graph/Op.hpp"

namespace mllm::ir {

IRContext::ptr_t trace_(nn::Module& module, const std::vector<Tensor>& ref_inputs, const std::vector<AnyValue>& args) {
  auto& ctx = Context::instance();
  auto this_session = ctx.thisThread();

  // Check if the Trace dispatcher is registered. If not register one for user.
  if (!ctx.dispatcherManager()->hasDispatcher(Dispatcher::trace_dispatcher_id)) {
    // Always need_async_exec_ false.
    ctx.dispatcherManager()->registerDispatcher(createIRTraceDispatcher(ctx.dispatcherManager()->getExecutor(), {}));
  }

  // Mark this thread SessionTCB as trace mode.
  this_session->trace_mode = true;

  // Create IRContext
  auto ir_context = std::make_shared<IRContext>();
  auto ir_module = ir_context->createAndSetModuleOp<ModuleOp>(ir_context->create<SymbolAttr>("main"));

  // Create init graph as the first graph.
  {
    auto init_graph = ir_context->create<::mllm::ir::graph::SubGraphOp>(ir_context->create<SymbolAttr>("init"));
    auto deinit_graph = ir_context->create<::mllm::ir::graph::SubGraphOp>(ir_context->create<SymbolAttr>("deinit"));
  }

  // Set current session's IRContext as temporary
  this_session->ir_context = ir_context;

  // Forward
  module.__trace(ref_inputs, args);

  // Mark this thread SessionTCB as normal mode.
  this_session->trace_mode = false;
  this_session->ir_context = nullptr;

  return ir_context;
}

namespace lowlevel {
void traceExternalValue() {
  // TODO
}

void traceComment(const std::string& comment) {
  auto& ctx = Context::instance();
  auto this_session = ctx.thisThread();
  auto ir_ctx = this_session->ir_context;
  ir_ctx->create<ir::dbg::CommentOp>("// " + comment);
}

void traceStart() {
  auto& ctx = Context::instance();
  auto this_session = ctx.thisThread();

  // Check if the Trace dispatcher is registered. If not register one for user.
  if (!ctx.dispatcherManager()->hasDispatcher(Dispatcher::trace_dispatcher_id)) {
    // Always need_async_exec_ false.
    ctx.dispatcherManager()->registerDispatcher(createIRTraceDispatcher(ctx.dispatcherManager()->getExecutor(), {}));
  }

  // Mark this thread SessionTCB as trace mode.
  this_session->trace_mode = true;

  // Create IRContext
  auto ir_context = std::make_shared<IRContext>();
  auto ir_module = ir_context->createAndSetModuleOp<ModuleOp>(ir_context->create<SymbolAttr>("main"));

  // Create init graph as the first graph.
  {
    auto init_graph = ir_context->create<::mllm::ir::graph::SubGraphOp>(ir_context->create<SymbolAttr>("init"));
    auto deinit_graph = ir_context->create<::mllm::ir::graph::SubGraphOp>(ir_context->create<SymbolAttr>("deinit"));
  }

  // Set current session's IRContext as temporary
  this_session->ir_context = ir_context;
}

IRContext::ptr_t traceStop() {
  auto& ctx = Context::instance();
  auto this_session = ctx.thisThread();

  auto ret = this_session->ir_context;

  // Mark this thread SessionTCB as normal mode.
  this_session->trace_mode = false;
  this_session->ir_context = nullptr;

  return ret;
}

void traceYield() {
  auto& ctx = Context::instance();
  auto this_session = ctx.thisThread();
  this_session->trace_mode = false;
}

void traceContinue() {
  auto& ctx = Context::instance();
  auto this_session = ctx.thisThread();
  this_session->trace_mode = true;
}
}  // namespace lowlevel
}  // namespace mllm::ir
