// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// This file implement a thread pool in high performance compute scenario. Especially, it is designed for CPU Ops.

#pragma once

#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>
#include <functional>
#include <condition_variable>

#define MLLM_HPC_THREAD_POOL_TASK_LIMITS 2

namespace mllm {

struct HpcThreadPoolTask {
  std::function<void(int)> func;
  int start = 0;
  int end = 0;
  int step = 1;

  // Things that will be modified by the thread pool
  bool __state_remap = false;
};

class HpcThreadPool {
 public:
  explicit HpcThreadPool(int thread_cnt = 0);

  ~HpcThreadPool();

  using ptr_t = std::shared_ptr<HpcThreadPool>;

  void push(HpcThreadPoolTask&& task, int task_slot_idx);

  void activate();

  void deactivate();

  void idle();

  void wakeup();

  int acquireTaskSlot();

  void releaseTaskSlot(int task_slot_idx);

  void bindCpuCore(std::vector<int>& cpu_mask);

  void showCpuTopology();

  void __threadPoolDestroy();

 private:
  void splitTask(HpcThreadPoolTask&& task, int task_slot_idx);

  int thread_cnt_ = 0;
  std::atomic_int available_task_slots_ = 0;
  std::atomic_int available_task_slots_old_ = 0;

  std::vector<std::thread> workers_;

  std::vector<bool> task_available_;
  std::vector<std::pair<HpcThreadPoolTask, std::vector<std::atomic_bool*>>> tasks_;

  std::atomic<bool> stop_ = false;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
};

}  // namespace mllm
