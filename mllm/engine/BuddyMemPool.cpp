// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <cmath>

#include "mllm/utils/Common.hpp"
#include "mllm/engine/BuddyMemPool.hpp"

namespace mllm {

namespace MLLM_ANONYMOUS_NAMESPACE {
size_t _log2_ceil(size_t size) {
  if (size == 0) return 0;
  return static_cast<size_t>(ceil(log2(size)));
}
}  // namespace MLLM_ANONYMOUS_NAMESPACE

BuddyMemPool::~BuddyMemPool() {
  for (auto& seg : context_.segments) { allocator_->generalFree(seg.second->ptr); }
}

BuddyMemPool::BuddyMemPool(BuddyMemPoolOptions options, const Allocator::ptr_t& allocator)
    : options_(std::move(options)), allocator_(allocator) {
  // align to 4KB
  void* _p = nullptr;
  allocator_->generalAlloc(&_p, options_.buddy_first_segment_cap, 4096);

  auto new_seg = new BuddyMemSegment{
      .ptr = (char*)_p,
      .cap = options_.buddy_first_segment_cap,
      .used = 0,
      .min_order = options_.buddy_min_order,
      .max_order = options_.buddy_max_order,
  };

  MLLM_RT_ASSERT_EQ(_log2_ceil(options_.buddy_first_segment_cap), options_.buddy_max_order);
  context_.segments.insert({new_seg->ptr, new_seg});
  context_.segment_blocks.insert(
      {new_seg->ptr, std::vector<std::list<BuddyMemBlock*>>(options_.buddy_max_order - options_.buddy_min_order + 1)});

  auto block = new BuddyMemBlock{
      .ptr = new_seg->ptr,
      .offset = 0,
      .size = options_.buddy_first_segment_cap,
      .segment = new_seg,
      .buddy_order = options_.buddy_max_order,
      .allocated = false,
  };

  context_.segment_blocks[new_seg->ptr][options_.buddy_max_order - options_.buddy_min_order].push_back(block);
}

void BuddyMemPool::alloc(Storage* s) {
  auto try_to_alloc_size = allocator_->allocSize(s);

  // Object cache
  BuddyMemBlock* omb = nullptr;
  if (options_.cache_small_obj_enabled && isInObjCache(try_to_alloc_size)) { omb = allocObjCache(try_to_alloc_size); }

  // Buddy
  if (!omb) { omb = allocBuddy(try_to_alloc_size); }
  s->ptr_ = omb->ptr;

  // Reg to memory pool
  st_.reg(s->custom_32bit_uuid_, omb);
}

void BuddyMemPool::alloc(const std::shared_ptr<Storage>& s) { alloc(s.get()); }

void BuddyMemPool::free(Storage* s) {
  auto try_to_alloc_size = allocator_->allocSize(s);

  if (s->ptr_ == nullptr) { return; }

  if (options_.cache_small_obj_enabled && isInObjCache(try_to_alloc_size)) {
    freeObjCache(st_[s->custom_32bit_uuid_]);
  } else {
    freeBuddy(st_[s->custom_32bit_uuid_]);
  }

  st_.remove(s->custom_32bit_uuid_);
}

void BuddyMemPool::free(const std::shared_ptr<Storage>& s) { free(s.get()); }

void BuddyMemPool::updateCacheSizeList(const std::unordered_set<size_t>& cache_size_set) {
  auto& object_cache = free_object_cache_st_;
  for (auto size : cache_size_set) {
    if (!isInObjCache(size)) {
      auto& free_list = object_cache[size];
      for (auto it : free_list) { freeBuddy(it); }
      object_cache.erase(object_cache.find(size));
    }
  }

  options_.cache_size_set = cache_size_set;
}

void BuddyMemPool::report() const {
  fmt::print("| Object Memory Hit Times: {:<27} |\n", obj_cache_hit_times_);
  fmt::print("| Object Memory Cached Times: {:<24} |\n", obj_cached_times_);
  fmt::print("+------------------------------------------------------+\n");
  for (auto& seg : context_.segments) {
    fmt::print("| address: {:#010x}, cap: {:>4}MB, used: {:>4}MB      |\n", (uintptr_t)seg.first,
               seg.second->cap / (1024 * 1024), seg.second->used / (1024 * 1024));
  }
  fmt::print("+------------------------------------------------------+\n");
}

BuddyMemBlock* BuddyMemPool::allocBuddy(size_t omb_size) {
  // lock_guard should in this scope.
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& cur_segs = context_.segments;
    auto& cur_seg_blocks = context_.segment_blocks;

    // loop all seg.
    for (auto seg : cur_segs) {
      auto& free_lists = cur_seg_blocks[seg.first];
      auto min_order = seg.second->min_order;
      auto max_order = seg.second->max_order;

      size_t required_size = std::max(omb_size, (size_t)(1ULL << min_order));
      if (required_size > (seg.second->cap - seg.second->used)) continue;

      size_t order = _log2_ceil(required_size);

      if (order > max_order) {
        MLLM_ERROR_EXIT(ExitCode::kMemory,
                        "The tensor size {} you want to alloc is too large. Buddy memory pool support max "
                        "tensor size is {}. You should change the `buddy_first_segment_cap` in MemManagerCargo.",
                        required_size, options_.buddy_first_segment_cap);
      }
      if (order < min_order) order = min_order;

      // search for usable order
      auto current_order = order;
      while (current_order <= max_order) {
        size_t idx = current_order - min_order;
        if (idx >= free_lists.size() || free_lists[idx].empty()) {
          current_order++;
          continue;
        }

        // get the empty block
        auto block = free_lists[idx].front();
        free_lists[idx].pop_front();

        // split this empty block if it has larger order
        while (block->buddy_order > order) {
          size_t new_size = block->size / 2;
          block->size = new_size;
          block->buddy_order--;

          // create block's buddy
          auto buddy = new BuddyMemBlock{
              .ptr = block->ptr + new_size,
              .offset = block->offset + new_size,
              .size = new_size,
              .segment = block->segment,
              .buddy_order = block->buddy_order,  // solver specific data
              .allocated = false,
          };

          free_lists[block->buddy_order - min_order].push_back(buddy);
        }

        block->allocated = true;

        seg.second->used += block->size;

        return block;
      }
    }
  }
  // lock_guard freed, so that we can call allocBuddy again if not found, alloc a new seg.
  if (expandBuddySegment()) { return allocBuddy(omb_size); }

  MLLM_ERROR_EXIT(ExitCode::kMemory, "Failed to alloc a new segment for buddy memory pool.");

  return nullptr;
}

void BuddyMemPool::freeBuddy(BuddyMemBlock* omb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto cur_seg = omb->segment;
  auto& cur_seg_blocks = context_.segment_blocks[cur_seg->ptr];

  size_t min_order = cur_seg->min_order;
  size_t max_order = cur_seg->max_order;

  omb->allocated = false;
  cur_seg->used -= omb->size;

  // joint buddy blocks
  while (true) {
    size_t current_order = omb->buddy_order;
    size_t buddy_size = (1ULL << current_order);

    auto buddy_addr = (size_t)(omb->ptr - cur_seg->ptr) ^ buddy_size;

    auto& list = cur_seg_blocks[current_order - min_order];
    auto it = std::find_if(list.begin(), list.end(), [&](const BuddyMemBlock* b) {
      return (b->ptr - cur_seg->ptr) == buddy_addr && b->buddy_order == current_order && !b->allocated;
    });

    if (it == list.end()) {
      list.push_back(omb);
      break;
    }

    if (omb->ptr < (*it)->ptr) {
      omb->size *= 2;
      omb->buddy_order++;
    } else {
      omb->ptr = (*it)->ptr;
      omb->size *= 2;
      omb->buddy_order++;
    }
    BuddyMemBlock* buddy = *it;
    list.erase(it);
    delete buddy;
  }
}

// This function is not thread safe, should be called in a thread safe context.
bool BuddyMemPool::expandBuddySegment() {
  auto& cur_segs = context_.segments;
  auto& cur_seg_blocks = context_.segment_blocks;

  size_t previous_seg_cap = 0;
  for (auto& seg : cur_segs) { previous_seg_cap = std::max(previous_seg_cap, (size_t)(1ULL << seg.second->max_order)); }

  size_t min_order = options_.buddy_min_order;
  size_t new_cap = std::min(previous_seg_cap * 2, (size_t)(1ULL << 29ULL));  // max is 512MB
  size_t max_order = _log2_ceil(new_cap);

  void* _p = nullptr;
  if (!allocator_->generalAlloc(&_p, new_cap, 4096)) { return false; }

  auto new_seg = new BuddyMemSegment{
      .ptr = (char*)_p,
      .cap = new_cap,
      .used = 0,
      .min_order = min_order,
      .max_order = max_order,
  };

  cur_segs.insert({new_seg->ptr, new_seg});
  cur_seg_blocks.insert({new_seg->ptr, std::vector<std::list<BuddyMemBlock*>>(max_order - min_order + 1)});

  auto block = new BuddyMemBlock{
      .ptr = new_seg->ptr,
      .offset = 0,
      .size = new_cap,
      .segment = new_seg,
      .buddy_order = max_order,
      .allocated = false,
  };
  cur_seg_blocks[new_seg->ptr][max_order - min_order].push_back(block);

  return true;
}

// This function is not thread safe, should be called in a thread safe context.
BuddyMemSegment* BuddyMemPool::locateSegment(char* ptr) {
  auto& segment = context_.segments;
  auto it = segment.upper_bound(ptr);
  if (it == segment.begin()) return nullptr;
  --it;
  auto seg = it->second;
  if (ptr >= seg->ptr && ptr < seg->ptr + seg->cap) { return seg; }
  return nullptr;
}

BuddyMemBlock* BuddyMemPool::allocObjCache(size_t omb_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& object_cache = free_object_cache_st_;

  if (!object_cache.count(omb_size) || object_cache[omb_size].empty()) return nullptr;

  auto ret = object_cache[omb_size].front();
  object_cache[omb_size].pop_front();

  obj_cache_hit_times_++;

  return ret;
}

void BuddyMemPool::freeObjCache(BuddyMemBlock* omb) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& object_cache = free_object_cache_st_;
  object_cache[omb->size].push_back(omb);
  obj_cached_times_++;
}

bool BuddyMemPool::isInObjCache(size_t omb_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  return options_.cache_size_set.count(omb_size);
}

}  // namespace mllm
