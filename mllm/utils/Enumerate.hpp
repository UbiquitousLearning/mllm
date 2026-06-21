// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace mllm {
namespace detail {

template<typename T_Index, typename T_Iterator>
class EnumerateResult {
 public:
  using IteratorRef = typename std::iterator_traits<T_Iterator>::reference;

  EnumerateResult(T_Index index, T_Iterator it) : index_(index), it_(it) {}

  T_Index get_index() const { return index_; }
  IteratorRef get_value() const { return *it_; }

 private:
  T_Index index_;
  T_Iterator it_;

 public:
  template<std::size_t N>
  decltype(auto) get() const {
    if constexpr (N == 0) {
      return get_index();
    } else if constexpr (N == 1) {
      return get_value();
    }
  }
};

template<typename T_Iterator>
class EnumerateIterator {
 public:
  using iterator_category = typename std::iterator_traits<T_Iterator>::iterator_category;
  using difference_type = typename std::iterator_traits<T_Iterator>::difference_type;
  using value_type = EnumerateResult<difference_type, T_Iterator>;
  using reference = value_type;
  using pointer = void;

  explicit EnumerateIterator(T_Iterator it, difference_type index = 0) : it_(it), index_(index) {}

  reference operator*() const { return reference(index_, it_); }

  EnumerateIterator& operator++() {
    ++it_;
    ++index_;
    return *this;
  }

  EnumerateIterator operator++(int) {
    EnumerateIterator temp = *this;
    ++(*this);
    return temp;
  }

  EnumerateIterator& operator--() {
    if constexpr (std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>) {
      --it_;
      --index_;
      return *this;
    } else {
      static_assert(std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>,
                    "operator-- is only available for bidirectional or random-access iterators.");
    }
  }

  EnumerateIterator operator--(int) {
    if constexpr (std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>) {
      EnumerateIterator temp = *this;
      --(*this);
      return temp;
    } else {
      static_assert(std::is_base_of_v<std::bidirectional_iterator_tag, iterator_category>,
                    "operator-- is only available for bidirectional or random-access iterators.");
    }
  }

  EnumerateIterator& operator+=(difference_type n) {
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iterator_category>) {
      it_ += n;
      index_ += n;
      return *this;
    } else {
      static_assert(std::is_base_of_v<std::random_access_iterator_tag, iterator_category>,
                    "operator+= is only available for random-access iterators.");
    }
  }

  EnumerateIterator& operator-=(difference_type n) {
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iterator_category>) {
      it_ -= n;
      index_ -= n;
      return *this;
    } else {
      static_assert(std::is_base_of_v<std::random_access_iterator_tag, iterator_category>,
                    "operator-= is only available for random-access iterators.");
    }
  }

  reference operator[](difference_type n) const {
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iterator_category>) {
      return *(*this + n);
    } else {
      static_assert(std::is_base_of_v<std::random_access_iterator_tag, iterator_category>,
                    "operator[] is only available for random-access iterators.");
    }
  }

  bool operator!=(const EnumerateIterator& other) const { return it_ != other.it_; }
  bool operator==(const EnumerateIterator& other) const { return it_ == other.it_; }
  bool operator<(const EnumerateIterator& other) const { return it_ < other.it_; }
  bool operator>(const EnumerateIterator& other) const { return it_ > other.it_; }
  bool operator<=(const EnumerateIterator& other) const { return it_ <= other.it_; }
  bool operator>=(const EnumerateIterator& other) const { return it_ >= other.it_; }

  template<typename U>
  friend auto operator+(EnumerateIterator<U> it, typename EnumerateIterator<U>::difference_type n) -> EnumerateIterator<U>;
  template<typename U>
  friend auto operator+(typename EnumerateIterator<U>::difference_type n, EnumerateIterator<U> it) -> EnumerateIterator<U>;
  template<typename U>
  friend auto operator-(EnumerateIterator<U> it, typename EnumerateIterator<U>::difference_type n) -> EnumerateIterator<U>;
  template<typename U>
  friend auto operator-(const EnumerateIterator<U>& lhs, const EnumerateIterator<U>& rhs) ->
      typename EnumerateIterator<U>::difference_type;

 private:
  T_Iterator it_;
  difference_type index_;
};

template<typename T_Iterator>
auto operator+(EnumerateIterator<T_Iterator> it,
               typename EnumerateIterator<T_Iterator>::difference_type n) -> EnumerateIterator<T_Iterator> {
  it += n;
  return it;
}
template<typename T_Iterator>
auto operator+(typename EnumerateIterator<T_Iterator>::difference_type n,
               EnumerateIterator<T_Iterator> it) -> EnumerateIterator<T_Iterator> {
  it += n;
  return it;
}
template<typename T_Iterator>
auto operator-(EnumerateIterator<T_Iterator> it,
               typename EnumerateIterator<T_Iterator>::difference_type n) -> EnumerateIterator<T_Iterator> {
  it -= n;
  return it;
}
template<typename T_Iterator>
auto operator-(const EnumerateIterator<T_Iterator>& lhs, const EnumerateIterator<T_Iterator>& rhs) ->
    typename EnumerateIterator<T_Iterator>::difference_type {
  return lhs.it_ - rhs.it_;
}

template<typename T_Iterable>
class EnumerateObject {
 public:
  explicit EnumerateObject(T_Iterable&& iterable) : iterable_(std::forward<T_Iterable>(iterable)) {}

  using iterator = decltype(std::begin(std::declval<T_Iterable&>()));
  using const_iterator = decltype(std::cbegin(std::declval<T_Iterable&>()));

  auto begin() { return EnumerateIterator(std::begin(iterable_)); }
  auto end() { return EnumerateIterator(std::end(iterable_)); }

  auto begin() const { return EnumerateIterator(std::cbegin(iterable_)); }
  auto end() const { return EnumerateIterator(std::cend(iterable_)); }
  auto cbegin() const { return EnumerateIterator(std::cbegin(iterable_)); }
  auto cend() const { return EnumerateIterator(std::cend(iterable_)); }

 private:
  T_Iterable iterable_;
};

template<typename T>
explicit EnumerateObject(T&&) -> EnumerateObject<T>;

}  // namespace detail

template<typename T_Iterable>
auto enumerate(T_Iterable&& iterable) {
  return detail::EnumerateObject(std::forward<T_Iterable>(iterable));
}

}  // namespace mllm

namespace std {
template<typename T_Index, typename T_Iterator>
struct tuple_size<mllm::detail::EnumerateResult<T_Index, T_Iterator>> : std::integral_constant<std::size_t, 2> {};

template<typename T_Index, typename T_Iterator>
struct tuple_element<0, mllm::detail::EnumerateResult<T_Index, T_Iterator>> {
  using type = T_Index;
};

template<typename T_Index, typename T_Iterator>
struct tuple_element<1, mllm::detail::EnumerateResult<T_Index, T_Iterator>> {
  using type = typename std::iterator_traits<T_Iterator>::reference;
};

template<std::size_t N, typename T_Index, typename T_Iterator>
decltype(auto) get(const mllm::detail::EnumerateResult<T_Index, T_Iterator>& result) {
  if constexpr (N == 0) {
    return result.get_index();
  } else if constexpr (N == 1) {
    return result.get_value();
  }
}
}  // namespace std
