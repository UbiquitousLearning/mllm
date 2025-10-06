// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <vector>
#include <memory>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/HaHaHash.hpp"

namespace mllm::prefix_cache {

//===----------------------------------------------------------------------===//
// VectorView
//
// NOTE:
// Radix Tree has many vector copy operations when split nodes, use VectorView
// to avoid copy.
//===----------------------------------------------------------------------===//
template<typename T>
class VectorView {
  static_assert(!std::is_const_v<T>, "T must not be const. Use const T explicitly if needed.");

 public:
  using value_type = T;
  using size_type = std::int32_t;
  using reference = T&;
  using const_reference = const T&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;

  VectorView() = default;

  explicit VectorView(const std::vector<T>& vec)
      : offset_(0), length_(static_cast<size_type>(vec.size())), data_(std::make_shared<std::vector<T>>(vec)) {}

  explicit VectorView(const std::shared_ptr<std::vector<T>> vec_ptr)
      : offset_(0), length_(vec_ptr ? static_cast<size_type>(vec_ptr->size()) : 0), data_(vec_ptr) {}

  VectorView(const std::vector<T>& vec, size_type off, size_type len)
      : VectorView(std::make_shared<std::vector<T>>(vec), off, len) {}

  VectorView(std::shared_ptr<std::vector<T>> vec_ptr, size_type off, size_type len) : data_(std::move(vec_ptr)) {
    if (!data_) {
      offset_ = 0;
      length_ = 0;
      return;
    }
    if (off < 0 || len < 0 || off + len > static_cast<size_type>(data_->size())) {
      throw std::out_of_range("VectorView: invalid offset/length");
    }
    offset_ = off;
    length_ = len;
  }

  reference at(size_type i) {
    check_bounds(i);
    return (*data_)[offset_ + i];
  }

  [[nodiscard]] const_reference at(size_type i) const {
    check_bounds(i);
    return (*data_)[offset_ + i];
  }

  reference operator[](size_type i) { return (*data_)[offset_ + i]; }

  const_reference operator[](size_type i) const { return (*data_)[offset_ + i]; }

  reference front() { return (*data_)[offset_]; }

  [[nodiscard]] const_reference front() const { return (*data_)[offset_]; }

  reference back() { return (*data_)[offset_ + length_ - 1]; }

  [[nodiscard]] const_reference back() const { return (*data_)[offset_ + length_ - 1]; }

  pointer data() { return data_->data() + offset_; }

  [[nodiscard]] const_pointer data() const { return data_->data() + offset_; }

  [[nodiscard]] size_type size() const noexcept { return length_; }

  [[nodiscard]] bool empty() const noexcept { return length_ == 0; }

  [[nodiscard]] size_type full_vector_size() const noexcept { return data_ ? static_cast<size_type>(data_->size()) : 0; }

  iterator begin() noexcept { return data(); }

  iterator end() noexcept { return data() + length_; }

  [[nodiscard]] const_iterator begin() const noexcept { return data(); }

  [[nodiscard]] const_iterator end() const noexcept { return data() + length_; }

  [[nodiscard]] const_iterator cbegin() const noexcept { return data(); }

  [[nodiscard]] const_iterator cend() const noexcept { return data() + length_; }

  [[nodiscard]] VectorView subview(size_type off, size_type len) const {
    if (off < 0 || len < 0 || off + len > length_) throw std::out_of_range("subview: out of range");
    return VectorView(data_, offset_ + off, len);
  }

  friend bool operator==(const VectorView& a, const VectorView& b) noexcept {
    if (a.size() != b.size()) { return false; }
    for (size_type i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) { return false; }
    }
    return true;
  }

  friend bool operator!=(const VectorView& a, const VectorView& b) noexcept { return !(a == b); }

 private:
  size_type offset_;
  size_type length_;
  std::shared_ptr<std::vector<T>> data_;

  void check_bounds(size_type i) const {
    if (i < 0 || i >= length_) throw std::out_of_range("VectorView::at");
  }
};

//===----------------------------------------------------------------------===//
// RadixTree Node Key
//===----------------------------------------------------------------------===//
struct RadixTreeNodeKey {
  VectorView<int64_t> token_ids;

  // LoRA id / Cache salt, etc.
  int64_t extra_key;

  RadixTreeNodeKey() = default;

  explicit RadixTreeNodeKey(const VectorView<int64_t>& token_ids_tmp, int64_t extra_key_tmp = 0);

  bool operator==(const RadixTreeNodeKey& o) const noexcept;

  bool operator!=(const RadixTreeNodeKey& o) const noexcept;
};

struct RadixTreeNodeKeyHash {
  inline size_t operator()(const RadixTreeNodeKey& k) const noexcept {
    XXH64_state_t* const state = XXH64_createState();
    if (!state) return 0;
    XXH64_reset(state, 0);
    if (!k.token_ids.empty()) { XXH64_update(state, k.token_ids.data(), k.token_ids.size() * sizeof(int64_t)); }
    XXH64_update(state, &k.extra_key, sizeof(int64_t));
    const std::uint64_t hash = XXH64_digest(state);
    XXH64_freeState(state);
    return static_cast<std::size_t>(hash);
  }
};

//===----------------------------------------------------------------------===//
// RadixTree Node Value
//===----------------------------------------------------------------------===//
struct RadixTreeNodeValue {
  // Layers[KV Caches]
  std::vector<VectorView<vp_addr_t>> k_cache_addresses;
  std::vector<VectorView<vp_addr_t>> v_cache_addresses;

  static inline RadixTreeNodeValue nil() { return RadixTreeNodeValue{}; }
};

//===----------------------------------------------------------------------===//
// RadixTree Node
//===----------------------------------------------------------------------===//
struct RadixTreeNode {
  // Key & Value
  RadixTreeNodeKey key;
  RadixTreeNodeValue value;

  RadixTreeNode* parent = nullptr;
  std::unordered_map<RadixTreeNodeKey, RadixTreeNode*, RadixTreeNodeKeyHash> children;

  // Control metadata
  int32_t ref_count = 0;  // Reference count for active users
  int32_t hit_count = 0;  // Hit count for LRU statistics
  std::chrono::steady_clock::time_point last_accessed{std::chrono::steady_clock::now()};

  RadixTreeNode();

  explicit RadixTreeNode(const RadixTreeNodeKey& k, const RadixTreeNodeValue& v, RadixTreeNode* p = nullptr);
};

//===----------------------------------------------------------------------===//
// RadixTree
//
// NOTE:
// 1. RadixTree just manages the index(vp_addr_t) of the cache.
// 2. GPU/CPU/NPU memory space is all in vp_addr_t(uint32_t) space
// 4. vp_addr_t(uint32_t) can represent 4G Tokens, which is enough for most(all, I guess) cases.
//===----------------------------------------------------------------------===//
struct RadixTreeOptions {
  bool enable_lru_eviction = true;  // Enable/disable LRU eviction
  float eviction_threshold = 0.9f;  // Evict when usage reaches this threshold

  bool enable_path_compression = false;  // Enable path compression
  size_t min_compression_length = 2;     // Minimum length for compression

  int32_t transformer_blocks_num = 1;  // Number of transformer blocks
};

struct RadixSearchResult {
  bool success = false;
  int32_t matched_length = 0;
  std::vector<std::pair<RadixTreeNode*, int32_t>> path;
  std::vector<std::vector<vp_addr_t>> k_cache_addresses;
  std::vector<std::vector<vp_addr_t>> v_cache_addresses;
};

class RadixTree {
 public:
  explicit RadixTree(const RadixTreeOptions& options = RadixTreeOptions{});

  ~RadixTree();

  void clear();

  // Only insert will split the node!
  bool insert(const RadixTreeNodeKey& key, const RadixTreeNodeValue& value);

  // Search will not split the node!
  RadixSearchResult search(const RadixTreeNodeKey& key);

  ///< "dot -Tpng tree.dot -o tree.png"
  [[nodiscard]] std::string dot() const;

 private:
  int32_t node_count_ = 0;
  RadixTreeOptions options_;
  std::unique_ptr<RadixTreeNode> root_ = nullptr;

  static inline void _dot(const RadixTreeNode* n, int32_t pid, int32_t& id_gen, std::ostringstream& os) {
    if (!n) return;
    int32_t my_id = id_gen++;
    os << "  n" << my_id << " [label=\"";
    for (auto t : n->key.token_ids) os << t << " ";
    os << "\\nref=" << n->ref_count << "\"];\n";
    if (pid >= 0) os << "  n" << pid << " -> n" << my_id << ";\n";
    for (auto& [k, c] : n->children) _dot(c, my_id, id_gen, os);
  }

  std::pair<RadixTreeNode*, RadixTreeNode*> _split(RadixTreeNode* node, size_t position);

  void _deleteNodeRecursive(RadixTreeNode* node);

  size_t _matchedLength(const RadixTreeNodeKey& cur_key, const RadixTreeNodeKey& pending_key);
};

}  // namespace mllm::prefix_cache
