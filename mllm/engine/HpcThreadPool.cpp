// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include "mllm/engine/HpcThreadPool.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm {

void HpcThreadPool::push(HpcThreadPoolTask&& task, int task_slot_idx) {
  int tiles_num = (task.end - task.start) / task.step;

  // There is no need to fall in back to thread pool
  if (tiles_num <= 1 || task_slot_idx < 0) {
    for (int i = task.start; i < task.end; i += task.step) { task.func(i); }
    return;
  }

  // Using thread pool
  splitTask(std::move(task), task_slot_idx);
}

void HpcThreadPool::activate() {
  {
    std::lock_guard<std::mutex> _l(queue_mutex_);
    available_task_slots_++;
    available_task_slots_old_++;
  }

  // Tell somebody we can work
  condition_.notify_all();
}

void HpcThreadPool::deactivate() {
  available_task_slots_--;
  available_task_slots_old_--;
}

void HpcThreadPool::idle() {
  std::lock_guard<std::mutex> _l(queue_mutex_);
  available_task_slots_old_ = available_task_slots_.load();
  available_task_slots_.store(0);
}

void HpcThreadPool::wakeup() {
  {
    std::lock_guard<std::mutex> _l(queue_mutex_);
    available_task_slots_.store(available_task_slots_old_);
  }
  condition_.notify_all();
}

int HpcThreadPool::acquireTaskSlot() {
  std::lock_guard<std::mutex> _l(queue_mutex_);
  for (int i = 0; i < MLLM_HPC_THREAD_POOL_TASK_LIMITS; ++i) {
    if (task_available_[i]) {
      task_available_[i] = false;
      return i;
    }
  }
  return -1;
}

void HpcThreadPool::releaseTaskSlot(int task_slot_idx) {
  if (task_slot_idx < 0 || task_slot_idx >= MLLM_HPC_THREAD_POOL_TASK_LIMITS) { return; }
  std::lock_guard<std::mutex> _l(queue_mutex_);
  task_available_[task_slot_idx] = true;
}

void HpcThreadPool::bindCpuCore(std::vector<int>& cpu_mask) {
  // TODO
}

void HpcThreadPool::showCpuTopology() {
  // TODO
}

void HpcThreadPool::__threadPoolDestroy() { available_task_slots_ = 0; }

void HpcThreadPool::splitTask(HpcThreadPoolTask&& task, int task_slot_idx) {
  // There are no task slots, use this main thread to compute!
  if (available_task_slots_ == 0) {
    for (int i = task.start; i < task.end; i += task.step) { task.func(i); }
    return;
  }
  int tiles_num = (task.end - task.start) / task.step;

  int _cnt = 0;
  std::vector<int> true_idx(tiles_num);
  for (int i = task.start; i < task.end; i += task.step) { true_idx[_cnt++] = i; }

  // When tiles_num > thread_cnt. We need to nested for loops. Which means static attach works to one thread.
  //
  // NOTE: Static dispatch may have performance issues in bit.LITTLE CPU Arch.
  //
  // NOTE: The dispatch logic is below:
  // e.g.: threads is 4, tiles_name is 12.
  //  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
  if (tiles_num > thread_cnt_) {
    tasks_[task_slot_idx].first = {
        .func =
            [tiles_num, &task, &true_idx, this](int thread_idx) {
              for (int v = thread_idx; v < tiles_num; v += thread_cnt_) { task.func(true_idx[v]); }
            },
        .start = 0,
        .end = thread_cnt_,
        .step = 1,
    };
    tiles_num = thread_cnt_;
  } else {
    tasks_[task_slot_idx].first = {
        .func = [tiles_num, &task, &true_idx, this](int thread_idx) { task.func(true_idx[thread_idx]); },
        .start = 0,
        .end = tiles_num,
        .step = 1,
    };
  }
  {
    for (int i = 1; i < tiles_num; ++i) { *tasks_[task_slot_idx].second[i] = true; }
  }

  // Explain why we start a 0 idx function.
  //
  // FIXME: The main thread is also used for compute or NOT ?
  tasks_[task_slot_idx].first.func(0);

  // Wait for all threads to complete
  bool complete = true;
  do {
    complete = true;
    for (int i = 1; i < tiles_num; ++i) {
      if (*tasks_[task_slot_idx].second[i]) {
        complete = false;
        break;
      }
    }
    std::this_thread::yield();
  } while (!complete);
}

HpcThreadPool::HpcThreadPool(int thread_cnt) {
  thread_cnt_ = thread_cnt;
  available_task_slots_ = 0;
  available_task_slots_old_ = 0;
  task_available_.resize(MLLM_HPC_THREAD_POOL_TASK_LIMITS);
  tasks_.resize(MLLM_HPC_THREAD_POOL_TASK_LIMITS);

  // Each task should hold some thread ok flag that mark this thread's work is done.
  for (int t = 0; t < MLLM_HPC_THREAD_POOL_TASK_LIMITS; ++t) {
    task_available_[t] = true;
    for (int i = 0; i < thread_cnt_; ++i) { tasks_[t].second.emplace_back(new std::atomic_bool{false}); }
  }

  for (int i = 1; i < thread_cnt_; ++i) {
    int thread_idx = i;
    workers_.emplace_back([this, thread_idx]() {
      while (!stop_) {
        while (available_task_slots_ > 0) {
          for (int i = 0; i < MLLM_HPC_THREAD_POOL_TASK_LIMITS; ++i) {
            if (*tasks_[i].second[thread_idx]) {
              tasks_[i].first.func(thread_idx);
              { *tasks_[i].second[thread_idx] = false; }
            }
          }
          std::this_thread::yield();
        }
        std::unique_lock<std::mutex> _l(queue_mutex_);
        condition_.wait(_l, [this] { return stop_ || available_task_slots_ > 0; });
      }
    });
  }
}

HpcThreadPool::~HpcThreadPool() {
  {
    std::lock_guard<std::mutex> _l(queue_mutex_);
    stop_ = true;
  }
  condition_.notify_all();
  for (auto& worker : workers_) { worker.join(); }
  for (auto& task : tasks_) {
    for (auto c : task.second) { delete c; }
  }
}

}  // namespace mllm
