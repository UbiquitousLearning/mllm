// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <list>
#include <atomic>
#include <thread>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "mllm/core/Storage.hpp"
#include "mllm/utils/SymbolTable.hpp"
#include "mllm/backends/base/Allocator.hpp"

namespace mllm {

struct BuddyMemSegment;

struct BuddyMemBlock {
  char* ptr = nullptr;
  size_t offset = 0;
  size_t size = 0;
  BuddyMemSegment* segment = nullptr;
  size_t buddy_order = 0;  // solver specific data
  bool allocated = false;
};

struct BuddyMemSegment {
  char* ptr = nullptr;
  size_t cap = 0;
  size_t used = 0;
  size_t min_order = 0;
  size_t max_order = 0;
};

struct BuddyMemPoolOptions {
  // Buddy related
  size_t buddy_first_segment_cap = 128 * 1024 * 1024;  // 128 MB
  size_t buddy_min_order = 14;                         // 16 KB
  size_t buddy_max_order = 27;                         // 128 MB

  // Cache size related
  bool cache_small_obj_enabled = false;
  std::unordered_set<size_t> cache_size_set;

  // Clean up periodically
  // If clean_up_period is 0, no clean up is performed.
  size_t clean_up_period = 0;  // ms
};

struct BuddyContext {
  std::map<char*, BuddyMemSegment*> segments;  // Sorted by ptr.
  std::map<char*, std::vector<std::list<BuddyMemBlock*>>> segment_blocks;
};

class BuddyMemPool {
 public:
  using ptr_t = std::shared_ptr<BuddyMemPool>;

  ~BuddyMemPool();

  explicit BuddyMemPool(BuddyMemPoolOptions options, const Allocator::ptr_t& allocator);

  void alloc(Storage* s);

  void alloc(const std::shared_ptr<Storage>& s);

  void free(Storage* s);

  void free(const std::shared_ptr<Storage>& s);

  void updateCacheSizeList(const std::unordered_set<size_t>& cache_size_set);

  void report() const;

 private:
  BuddyMemBlock* allocBuddy(size_t omb_size);

  void freeBuddy(BuddyMemBlock* omb);

  // This function is not thread safe, should be called in a thread safe context.
  bool expandBuddySegment();

  // This function is not thread safe, should be called in a thread safe context.
  BuddyMemSegment* locateSegment(char* ptr);

  BuddyMemBlock* allocObjCache(size_t omb_size);

  void freeObjCache(BuddyMemBlock* omb);

  bool isInObjCache(size_t omb_size);

  Allocator::ptr_t allocator_;
  BuddyMemPoolOptions options_;

  BuddyContext context_;
  SymbolTable<uint32_t, BuddyMemBlock*> st_;

  int32_t obj_cache_hit_times_ = 0;
  int32_t obj_cached_times_ = 0;
  std::unordered_map<size_t, std::list<BuddyMemBlock*>> free_object_cache_st_;

  // Make manager cleanup thread safe.
  std::mutex mutex_;
  std::atomic<bool> cleanup_thread_running_{false};
  std::thread cleanup_thread_;
};

}  // namespace mllm
