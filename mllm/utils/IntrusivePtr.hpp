// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <atomic>
#include <cassert>
#include <utility>
#include <cstdint>

namespace mllm {

template<typename T>
class IntrusivePtr;

class RefCountedBase {
 public:
  RefCountedBase() noexcept : ref_count(0) {}
  virtual ~RefCountedBase() = default;

  RefCountedBase(const RefCountedBase&) = delete;
  RefCountedBase& operator=(const RefCountedBase&) = delete;

  void intrusive_ptr_add_ref() noexcept { ref_count.fetch_add(1, std::memory_order_relaxed); }

  void intrusive_ptr_release() noexcept {
    if (ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) { delete this; }
  }

  int64_t use_count() const noexcept { return ref_count.load(std::memory_order_acquire); }

 private:
  std::atomic<int64_t> ref_count;
};

template<typename T>
class IntrusivePtr {
 public:
  using element_type = T;

  IntrusivePtr() noexcept : ptr(nullptr) {}

  explicit IntrusivePtr(T* p) noexcept : ptr(p) {
    if (p) { p->intrusive_ptr_add_ref(); }
  }

  IntrusivePtr(const IntrusivePtr& other) noexcept : ptr(other.ptr) {
    if (ptr) { ptr->intrusive_ptr_add_ref(); }
  }

  IntrusivePtr(IntrusivePtr&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }

  template<typename U>
  IntrusivePtr(const IntrusivePtr<U>& other) noexcept : ptr(other.get()) {  // NOLINT
    if (ptr) { ptr->intrusive_ptr_add_ref(); }
  }

  template<typename U>
  IntrusivePtr(IntrusivePtr<U>&& other) noexcept : ptr(other.get()) {  // NOLINT
    other.reset();
  }

  ~IntrusivePtr() {
    if (ptr) { ptr->intrusive_ptr_release(); }
  }

  IntrusivePtr& operator=(const IntrusivePtr& other) noexcept {
    IntrusivePtr(other).swap(*this);
    return *this;
  }

  IntrusivePtr& operator=(IntrusivePtr&& other) noexcept {
    IntrusivePtr(std::move(other)).swap(*this);
    return *this;
  }

  template<typename U>
  IntrusivePtr& operator=(const IntrusivePtr<U>& other) noexcept {
    IntrusivePtr(other).swap(*this);
    return *this;
  }

  template<typename U>
  IntrusivePtr& operator=(IntrusivePtr<U>&& other) noexcept {
    IntrusivePtr(std::move(other)).swap(*this);
    return *this;
  }

  IntrusivePtr& operator=(T* p) noexcept {
    IntrusivePtr(p).swap(*this);
    return *this;
  }

  void reset() noexcept { IntrusivePtr().swap(*this); }

  void reset(T* p) noexcept { IntrusivePtr(p).swap(*this); }

  void swap(IntrusivePtr& other) noexcept { std::swap(ptr, other.ptr); }

  T* get() const noexcept { return ptr; }

  T& operator*() const noexcept {
    assert(ptr != nullptr);
    return *ptr;
  }

  T* operator->() const noexcept {
    assert(ptr != nullptr);
    return ptr;
  }

  explicit operator bool() const noexcept { return ptr != nullptr; }

  [[nodiscard]] int64_t use_count() const noexcept { return ptr ? ptr->use_count() : 0; }

  bool operator==(const IntrusivePtr& other) const noexcept { return ptr == other.ptr; }

  bool operator!=(const IntrusivePtr& other) const noexcept { return ptr != other.ptr; }

  bool operator<(const IntrusivePtr& other) const noexcept { return ptr < other.ptr; }

  template<typename U>
  friend class IntrusivePtr;

 private:
  T* ptr;
};

template<typename T>
void swap(IntrusivePtr<T>& lhs, IntrusivePtr<T>& rhs) noexcept {
  lhs.swap(rhs);
}

template<typename T, typename U>
bool operator==(const IntrusivePtr<T>& lhs, const IntrusivePtr<U>& rhs) noexcept {
  return lhs.get() == rhs.get();
}

template<typename T, typename U>
bool operator!=(const IntrusivePtr<T>& lhs, const IntrusivePtr<U>& rhs) noexcept {
  return lhs.get() != rhs.get();
}

template<typename T, typename U>
bool operator<(const IntrusivePtr<T>& lhs, const IntrusivePtr<U>& rhs) noexcept {
  return lhs.get() < rhs.get();
}

template<typename T, typename... Args>
IntrusivePtr<T> make_intrusive(Args&&... args) {
  return IntrusivePtr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace mllm
