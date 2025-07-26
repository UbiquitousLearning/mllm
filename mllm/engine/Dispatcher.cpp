/**
 * @file Dispatcher.cpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-25
 *
 */
#include "mllm/engine/Dispatcher.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

Dispatcher::Dispatcher(exec::static_thread_pool& thread_pool, dispatcher_id_t id)
    : thread_pool_(thread_pool), dispatcher_id_(id) {}

void Dispatcher::receive(const Task::ptr_t& task) { NYI("Dispatcher::receive is not implemented"); }

void Dispatcher::process(const Task::ptr_t& task) { NYI("Dispatcher::process is not implemented"); }

void Dispatcher::syncWait() { NYI("Dispatcher::syncWait is not implemented"); }

}  // namespace mllm