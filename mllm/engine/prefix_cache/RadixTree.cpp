// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/utils/Common.hpp"
#include "mllm/engine/prefix_cache/RadixTree.hpp"

namespace mllm::prefix_cache {
//===----------------------------------------------------------------------===//
// RadixTree Node Key
//===----------------------------------------------------------------------===//
RadixTreeNodeKey::RadixTreeNodeKey(const VectorView<int64_t>& token_ids_tmp, int64_t extra_key_tmp)
    : token_ids(token_ids_tmp), extra_key(extra_key_tmp) {}

bool RadixTreeNodeKey::operator==(const RadixTreeNodeKey& o) const noexcept { return token_ids == o.token_ids; }

bool RadixTreeNodeKey::operator!=(const RadixTreeNodeKey& o) const noexcept { return !(*this == o); }

//===----------------------------------------------------------------------===//
// RadixTree Node
//===----------------------------------------------------------------------===//
RadixTreeNode::RadixTreeNode() = default;

RadixTreeNode::RadixTreeNode(const RadixTreeNodeKey& k, const RadixTreeNodeValue& v, RadixTreeNode* p)
    : key(k), value(v), parent(p) {}

//===----------------------------------------------------------------------===//
// RadixTree
//===----------------------------------------------------------------------===//
RadixTree::RadixTree(const RadixTreeOptions& options) : options_(options), root_(std::make_unique<RadixTreeNode>()) {
  std::vector<int64_t> root_tokens{-1};  // Root sentinel
  root_->key = RadixTreeNodeKey(VectorView{root_tokens});
  node_count_ += 1;
}

RadixTree::~RadixTree() { clear(); }

void RadixTree::clear() {
  for (auto& [key, child] : root_->children) { _deleteNodeRecursive(child); }
  root_->children.clear();
  root_->parent = nullptr;
  node_count_ = 1;  // Only root remains
}

bool RadixTree::insert(const RadixTreeNodeKey& key, const RadixTreeNodeValue& value) {
  // CASE 1: empty tree, seq[1, 2, 3] -> just insert
  // CASE 2: seq[1, 2, 3, 4] exists, want to insert seq[1, 2, 3] -> split seq[1, 2, 3] + seq[4]
  // CASE 3. seq[1, 2, 3] exists, want to insert seq[1, 2, 3, 4, 5] -> split seq[1, 2, 3] + seq[4, 5]
  // CASE 4. seq[3, 4, 5] exists, want to insert seq[2, 3, 4] -> split to seq[2] + seq[3, 4] + seq[5]
  // CASE 5. seq[6, 7] exists, want to insert seq[1, 2] -> jut insert it.
  if (key.token_ids.empty()) { return true; }

  RadixTreeNode* cur = root_.get();
  size_t matched = 0;
  RadixTreeNode* deepest = cur;  // deepest node
  size_t edge_matched = 0;       // edge_matched on deepest node

  while (matched < key.token_ids.size() && cur) {
    VectorView<int64_t> rest = key.token_ids.subview(matched, key.token_ids.size() - matched);
    RadixTreeNodeKey rest_key{rest, key.extra_key};

    bool found = false;
    for (auto& [child_key, child_ptr] : cur->children) {
      size_t ml = _matchedLength(rest_key, child_key);
      if (ml == 0) continue;
      deepest = child_ptr;
      edge_matched = ml;
      matched += ml;
      found = true;
      break;
    }
    if (!found) break;
    cur = deepest;
  }

  if (matched == key.token_ids.size()) {
    deepest->ref_count++;
    return true;
  }

  // CASE like: seq[1, 2, 3, 4] exists, want to insert seq[1, 2, 3, 7]
  if (matched < key.token_ids.size() && edge_matched > 0 && edge_matched < deepest->key.token_ids.size()) {
    auto [upper, lower] = _split(deepest, edge_matched);
    (void)lower;
    deepest = upper;
  }

  cur = deepest;

  // Other CASE
  auto suffix = key.token_ids.subview(matched, key.token_ids.size() - matched);
  auto leaf_key = RadixTreeNodeKey(suffix, key.extra_key);
  auto leaf_value = RadixTreeNodeValue();
  leaf_value.k_cache_addresses.resize(options_.transformer_blocks_num);
  leaf_value.v_cache_addresses.resize(options_.transformer_blocks_num);
  for (int b_idx = 0; b_idx < options_.transformer_blocks_num; ++b_idx) {
    leaf_value.k_cache_addresses[b_idx] = value.k_cache_addresses[b_idx].subview(matched, key.token_ids.size() - matched);
    leaf_value.v_cache_addresses[b_idx] = value.v_cache_addresses[b_idx].subview(matched, key.token_ids.size() - matched);
  }
  RadixTreeNode* leaf = new RadixTreeNode(leaf_key, leaf_value, cur);
  cur->children.emplace(leaf_key, leaf);
  node_count_++;
  return true;
}

RadixSearchResult RadixTree::search(const RadixTreeNodeKey& key) {
  std::vector<std::pair<RadixTreeNode*, int32_t>> path;

  RadixSearchResult result;
  result.success = false;
  result.matched_length = 0;
  result.k_cache_addresses = {};
  result.v_cache_addresses = {};

  RadixTreeNode* cur_node = root_.get();
  size_t cur_searched_len = 0;
  path.emplace_back(cur_node, 0);

  // Start from root to find.
  while (cur_searched_len < key.token_ids.size() && cur_node) {
    bool found_next_child = false;
    auto sub_token_ids = key.token_ids.subview(cur_searched_len, key.token_ids.size() - cur_searched_len);

    for (auto& [child_key, child_node] : cur_node->children) {
      // Find how many tokens matched in this node.
      auto matched_len = _matchedLength(RadixTreeNodeKey{sub_token_ids, key.extra_key}, child_key);

      if (matched_len) {
        // Update loop state
        cur_node = child_node;
        cur_searched_len += matched_len;
        path.emplace_back(cur_node, matched_len);
        found_next_child = true;
        break;
      }
    }

    if (!found_next_child) { break; }
  }

  if (cur_searched_len) {
    result.success = true;
    result.path = path;
    result.matched_length = cur_searched_len;
    result.k_cache_addresses.resize(options_.transformer_blocks_num);
    result.v_cache_addresses.resize(options_.transformer_blocks_num);

    // Flatten path in result.
    for (auto& [node, len] : path) {
      for (int b_idx = 0; b_idx < options_.transformer_blocks_num; ++b_idx) {
        for (int e_idx = 0; e_idx < len; ++e_idx) {
          result.k_cache_addresses[b_idx].push_back(node->value.k_cache_addresses[b_idx][e_idx]);
          result.v_cache_addresses[b_idx].push_back(node->value.v_cache_addresses[b_idx][e_idx]);
        }
      }
    }
  } else {
    result.success = false;
    result.path = path;
    result.matched_length = 0;
    result.k_cache_addresses.resize(options_.transformer_blocks_num);
    result.v_cache_addresses.resize(options_.transformer_blocks_num);
  }

  return result;
}

std::string RadixTree::dot() const {
  std::ostringstream os;
  os << "digraph Radix {\n"
     << "  node [shape=box, fontname=\"Mono\"];\n"
     << "  edge [arrowhead=vee];\n";
  int32_t id_cnt = 0;
  _dot(root_.get(), -1, id_cnt, os);
  os << "}\n";
  return os.str();
}

std::pair<RadixTreeNode*, RadixTreeNode*> RadixTree::_split(RadixTreeNode* node, size_t position) {
  const size_t old_sz = node->key.token_ids.size();
  if (position == 0) return {nullptr, node};
  if (position >= old_sz) return {node, nullptr};

  // original node split to [upper_node, lower_node]
  // upper_node --> lower_node

  // Processing lower_node
  {
    auto lower_token_ids = node->key.token_ids.subview(position, old_sz - position);
    RadixTreeNodeKey lower_key(lower_token_ids, node->key.extra_key);
    RadixTreeNodeValue lower_value;
    lower_value.k_cache_addresses.resize(options_.transformer_blocks_num);
    lower_value.v_cache_addresses.resize(options_.transformer_blocks_num);
    for (int b_idx = 0; b_idx < options_.transformer_blocks_num; ++b_idx) {
      lower_value.k_cache_addresses[b_idx] = node->value.k_cache_addresses[b_idx].subview(position, old_sz - position);
      lower_value.v_cache_addresses[b_idx] = node->value.v_cache_addresses[b_idx].subview(position, old_sz - position);
    }

    // Parent is node.
    auto lower = new RadixTreeNode(lower_key, lower_value, node);

    // Move all children from upper yo lower
    lower->children.swap(node->children);
    for (auto& [k, c] : lower->children) c->parent = lower;

    // node get lower
    node->children.emplace(lower_key, lower);

    // update upper node
    auto old_node_key = node->key;
    node->key.token_ids = node->key.token_ids.subview(0, position);
    for (int b = 0; b < options_.transformer_blocks_num; ++b) {
      node->value.k_cache_addresses[b] = node->value.k_cache_addresses[b].subview(0, position);
      node->value.v_cache_addresses[b] = node->value.v_cache_addresses[b].subview(0, position);
    }
    node_count_++;

    // upper node's key is changed, we need modify it's parent children scope
    if (node->parent) {
      auto& parent_children = node->parent->children;
      auto it = parent_children.find(old_node_key);
      if (it != parent_children.end()) {
        parent_children.erase(it);
        parent_children.emplace(node->key, node);
      } else {
        MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't find node in parent children");
      }
    }

    return {node, lower};
  }
}

void RadixTree::_deleteNodeRecursive(RadixTreeNode* node) {
  if (!node) return;
  // Delete all child.
  for (auto& [key, child] : node->children) { _deleteNodeRecursive(child); }
  // Delete parent.
  delete node;
}

size_t RadixTree::_matchedLength(const RadixTreeNodeKey& cur_key, const RadixTreeNodeKey& pending_key) {
  if (cur_key.extra_key != pending_key.extra_key) { return 0; }
  size_t n = std::min(cur_key.token_ids.size(), pending_key.token_ids.size());
  for (size_t i = 0; i < n; ++i) {
    if (cur_key.token_ids[i] != pending_key.token_ids[i]) return i;
  }
  return n;
}

}  // namespace mllm::prefix_cache
