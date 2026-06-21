// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <typeinfo>
#include <utility>

namespace mllm {

class bad_any_value_cast : public std::bad_cast {
 public:
  [[nodiscard]] const char* what() const noexcept override { return "bad any_value_cast: failed type conversion"; }
};

struct copy_t {};
inline constexpr copy_t any_copy_tag;

class AnyValue {
 private:
  struct Manager_ {
    void (*deleter)(void*);
    void* (*cloner)(const void*);
  };

  void* data_ = nullptr;
  const std::type_info* type_info_ = &typeid(void);
  const Manager_* manager_ = nullptr;

 public:
  AnyValue() noexcept = default;

  ~AnyValue() { reset(); }

  template<typename T, typename DecayedT = std::decay_t<T>>
  explicit AnyValue(T& value) noexcept
    requires(!std::is_same_v<DecayedT, AnyValue>)
  {
    data_ = &value;
    type_info_ = &typeid(T);
    manager_ = nullptr;
  }

  template<typename T>
  AnyValue(copy_t, const T& value) {
    using OwnedType = std::decay_t<T>;
    data_ = new OwnedType(value);
    type_info_ = &typeid(T);
    manager_ = manager_for_<OwnedType>();
  }

  template<typename T, typename DecayedT = std::decay_t<T>>
  explicit AnyValue(T&& value) noexcept
    requires(!std::is_same_v<DecayedT, AnyValue> && !std::is_const_v<T> && !std::is_reference_v<T>)
  {
    using OwnedType = DecayedT;
    data_ = new OwnedType(std::move(value));
    type_info_ = &typeid(T);
    manager_ = manager_for_<OwnedType>();
  }

  AnyValue(const AnyValue& other) {
    if (other.manager_) {
      data_ = other.manager_->cloner(other.data_);
      type_info_ = other.type_info_;
      manager_ = other.manager_;
    } else {
      data_ = other.data_;
      type_info_ = other.type_info_;
      manager_ = nullptr;
    }
  }

  AnyValue(AnyValue&& other) noexcept : data_(other.data_), type_info_(other.type_info_), manager_(other.manager_) {
    other.data_ = nullptr;
    other.type_info_ = &typeid(void);
    other.manager_ = nullptr;
  }

  AnyValue& operator=(const AnyValue& other) {
    AnyValue(other).swap(*this);
    return *this;
  }

  AnyValue& operator=(AnyValue&& other) noexcept {
    AnyValue(std::move(other)).swap(*this);
    return *this;
  }

  template<typename T, typename DecayedT = std::decay_t<T>>
  AnyValue& operator=(T&& value)
    requires(!std::is_same_v<DecayedT, AnyValue>)
  {
    emplace<DecayedT>(std::forward<T>(value));
    return *this;
  }

  void reset() noexcept {
    if (manager_) { manager_->deleter(data_); }
    data_ = nullptr;
    type_info_ = &typeid(void);
    manager_ = nullptr;
  }

  template<typename T, typename... Args>
  T& emplace(Args&&... args) {
    reset();
    using OwnedType = std::decay_t<T>;
    data_ = new OwnedType(std::forward<Args>(args)...);
    type_info_ = &typeid(T);
    manager_ = manager_for_<OwnedType>();
    return *static_cast<OwnedType*>(data_);
  }

  void swap(AnyValue& other) noexcept {
    std::swap(data_, other.data_);
    std::swap(type_info_, other.type_info_);
    std::swap(manager_, other.manager_);
  }

  [[nodiscard]] bool has_value() const noexcept { return data_ != nullptr; }

  [[nodiscard]] bool is_reference() const noexcept { return has_value() && (manager_ == nullptr); }

  [[nodiscard]] const std::type_info& type() const noexcept { return *type_info_; }

  template<typename T>
  T& get() {
    if (typeid(T) != *type_info_) { throw bad_any_value_cast(); }
    return *static_cast<T*>(data_);
  }

  template<typename T>
  const T& get() const {
    if (typeid(T) != *type_info_) { throw bad_any_value_cast(); }
    return *static_cast<const T*>(data_);
  }

  template<typename T>
  T* get_if() noexcept {
    return (typeid(T) == *type_info_) ? static_cast<T*>(data_) : nullptr;
  }

  template<typename T>
  const T* get_if() const noexcept {
    return (typeid(T) == *type_info_) ? static_cast<const T*>(data_) : nullptr;
  }

 private:
  template<typename T>
  static const Manager_* manager_for_() noexcept {
    static constexpr Manager_ manager = {
        /* deleter */ [](void* data) { delete static_cast<T*>(data); },
        /* cloner */ [](const void* data) -> void* { return new T(*static_cast<const T*>(data)); }};
    return &manager;
  }
};

}  // namespace mllm
