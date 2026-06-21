// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

Dispatcher::Dispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id)
    : thread_pool_(thread_pool), dispatcher_id_(id) {}

void Dispatcher::setPreprocessTaskFunc(const preprocess_task_func_t& func) { preprocess_task_func_ = func; }

void Dispatcher::setAfterprocessTaskFunc(const afterprocess_task_func_t& func) { afterprocess_task_func_ = func; }

void Dispatcher::receive(const Task::ptr_t& task) { NYI("Dispatcher::receive is not implemented"); }

void Dispatcher::process(const Task::ptr_t& task) { NYI("Dispatcher::process is not implemented"); }

void Dispatcher::syncWait() { NYI("Dispatcher::syncWait is not implemented"); }

void Dispatcher::preprocessTask(const Task::ptr_t& task) { preprocess_task_func_(task); };

void Dispatcher::afterprocessTask(const Task::ptr_t& task) { afterprocess_task_func_(task); };

}  // namespace mllm