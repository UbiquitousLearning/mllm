// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>

namespace mllm {

template<typename T>
class WeakOwner {
 public:
  WeakOwner() noexcept = default;

  WeakOwner(std::nullptr_t) noexcept {}  // NOLINT

  template<typename U>
  WeakOwner(const std::shared_ptr<U>& shared) noexcept : ptr_(shared.get()) {}  // NOLINT

  template<typename U>
  WeakOwner(std::shared_ptr<U>&& shared) noexcept : ptr_(shared.get()) {}  // NOLINT

  WeakOwner(const WeakOwner& other) noexcept = default;

  WeakOwner(WeakOwner&& other) noexcept = default;

  template<typename U>
  WeakOwner(const WeakOwner<U>& other) noexcept  // NOLINT
    requires(std::is_convertible_v<U*, T*>)
      : ptr_(other.get_weak()) {}

  ~WeakOwner() = default;

  WeakOwner& operator=(const WeakOwner& other) noexcept = default;

  WeakOwner& operator=(WeakOwner&& other) noexcept = default;

  template<typename U>
  WeakOwner& operator=(const std::shared_ptr<U>& shared) noexcept {
    ptr_ = shared.get();
    return *this;
  }

  template<typename U>
  WeakOwner& operator=(std::shared_ptr<U>&& shared) noexcept {
    ptr_ = shared.get();
    return *this;
  }

  WeakOwner& operator=(std::nullptr_t) noexcept {
    ptr_ = nullptr;
    return *this;
  }

  T* get_weak() const noexcept { return ptr_; }

  [[nodiscard]] bool expired() const noexcept { return ptr_ == nullptr; }

  void reset() noexcept { ptr_ = nullptr; }

  T* operator->() { return ptr_; }

  T& operator*() { return *ptr_; }

  template<typename U>
  bool operator==(const WeakOwner<U>& other) const noexcept {
    return ptr_ == other.get_weak();
  }

  template<typename U>
  bool operator==(const std::weak_ptr<U>& other) const noexcept {
    return ptr_ == other.get_weak();
  }

  template<typename U>
  bool operator==(const std::shared_ptr<U>& other) const noexcept {
    return ptr_ == other.get();
  }

  bool operator==(std::nullptr_t) const noexcept { return expired(); }

  template<typename U>
  bool operator!=(const WeakOwner<U>& other) const noexcept {
    return !(*this == other);
  }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

 private:
  T* ptr_ = nullptr;
};

}  // namespace mllm
