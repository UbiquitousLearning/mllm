// RADIX TREE TEST
//
// Run:
// for i in {1..6}; do dot -Tpng case$i.dot -o case$i.png; done
// in shell to check.
#include <random>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>

#include "mllm/utils/Common.hpp"
#include "mllm/engine/prefix_cache/RadixTree.hpp"

using namespace mllm::prefix_cache;  // NOLINT

static RadixTreeNodeKey makeKey(std::initializer_list<int64_t> il) {
  return RadixTreeNodeKey(VectorView<int64_t>(std::vector<int64_t>(il)));
}

static RadixTreeNodeValue makeValue(size_t len, int blocks) {
  RadixTreeNodeValue v;
  v.k_cache_addresses.resize(blocks);
  v.v_cache_addresses.resize(blocks);

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<vp_addr_t> dist;

  for (int b = 0; b < blocks; ++b) {
    std::vector<vp_addr_t> tmp;
    tmp.reserve(len);
    for (size_t i = 0; i < len; ++i) tmp.push_back(dist(rng));

    v.k_cache_addresses[b] = VectorView<vp_addr_t>(tmp);
    v.v_cache_addresses[b] = VectorView<vp_addr_t>(tmp);
  }
  return v;
}

static void save_dot(const RadixTree& tree, const char* file) { std::ofstream(file) << tree.dot(); }

/* ---------------- 6 CASE ---------------- */
int main() {
  RadixTreeOptions opt;
  opt.transformer_blocks_num = 2;

  /*-------- CASE 1  empty tree  insert 1,2,3  --------*/
  {
    RadixTree tree(opt);
    auto value = makeValue(3, opt.transformer_blocks_num);
    tree.insert(makeKey({1, 2, 3}), value);

    auto result = tree.search(makeKey({1, 2, 3}));
    MLLM_RT_ASSERT(result.success);
    for (int layer = 0; layer < opt.transformer_blocks_num; ++layer) {
      for (int i = 0; i < 3; ++i) {
        MLLM_RT_ASSERT_EQ(result.k_cache_addresses[layer][i], value.k_cache_addresses[layer][i]);
        MLLM_RT_ASSERT_EQ(result.v_cache_addresses[layer][i], value.v_cache_addresses[layer][i]);
      }
    }

    save_dot(tree, "case1.dot");
  }

  /*-------- CASE 2  1,2,3,4 exists  insert 1,2,3 --------*/
  // This case will not appear in llm service, but we need to test it.
  {
    RadixTree tree(opt);
    tree.insert(makeKey({1, 2, 3, 4}), makeValue(4, opt.transformer_blocks_num));
    tree.insert(makeKey({1, 2, 3}), makeValue(3, opt.transformer_blocks_num));
    save_dot(tree, "case2.dot");
  }

  /*-------- CASE 3  1,2,3 exists  insert 1,2,3,4,5 --------*/
  {
    RadixTree tree(opt);
    tree.insert(makeKey({1, 2, 3}), makeValue(3, opt.transformer_blocks_num));
    tree.insert(makeKey({1, 2, 3, 4, 5}), makeValue(5, opt.transformer_blocks_num));
    save_dot(tree, "case3.dot");
  }

  /*-------- CASE 4  3,4,5 exists  insert 2,3,4 --------*/
  {
    RadixTree tree(opt);
    tree.insert(makeKey({3, 4, 5}), makeValue(3, opt.transformer_blocks_num));
    tree.insert(makeKey({2, 3, 4}), makeValue(3, opt.transformer_blocks_num));
    save_dot(tree, "case4.dot");
  }

  /*-------- CASE 5  6,7 exists  insert 1,2 --------*/
  {
    RadixTree tree(opt);
    tree.insert(makeKey({6, 7}), makeValue(2, opt.transformer_blocks_num));
    tree.insert(makeKey({1, 2}), makeValue(2, opt.transformer_blocks_num));
    save_dot(tree, "case5.dot");
  }

  /*-------- CASE 6  1,2,3,4 exists  insert 1,2,3,7 --------*/
  {
    RadixTree tree(opt);
    auto value0 = makeValue(4, opt.transformer_blocks_num);
    auto value1 = makeValue(4, opt.transformer_blocks_num);
    tree.insert(makeKey({1, 2, 3, 4}), value0);
    tree.insert(makeKey({1, 2, 3, 7}), value1);

    auto result0 = tree.search(makeKey({1, 2, 3, 4}));
    auto result1 = tree.search(makeKey({1, 2, 3, 7}));

    MLLM_RT_ASSERT(result0.success);
    MLLM_RT_ASSERT(result1.success);
    for (int layer = 0; layer < opt.transformer_blocks_num; ++layer) {
      for (int i = 0; i < 4; ++i) {
        MLLM_RT_ASSERT_EQ(result0.k_cache_addresses[layer][i], value0.k_cache_addresses[layer][i]);
        MLLM_RT_ASSERT_EQ(result0.v_cache_addresses[layer][i], value0.v_cache_addresses[layer][i])
      }
    }

    for (int layer = 0; layer < opt.transformer_blocks_num; ++layer) {
      for (int i = 0; i < 3; ++i) {
        MLLM_RT_ASSERT_EQ(result1.k_cache_addresses[layer][i], value0.k_cache_addresses[layer][i]);
        MLLM_RT_ASSERT_EQ(result1.v_cache_addresses[layer][i], value0.v_cache_addresses[layer][i])
      }
      MLLM_RT_ASSERT_EQ(result1.k_cache_addresses[layer][3], value1.k_cache_addresses[layer][3]);
    }
    save_dot(tree, "case6.dot");
  }

  printf("ALL CASE PASS!\n");
  return 0;
}
