#include "mllm/mllm.hpp"
#include "mllm/engine/prefix_cache/TLB.hpp"
#include "mllm/engine/prefix_cache/Cache.hpp"

MLLM_MAIN({
  mllm::prefix_cache::CacheOptions opt{.radix_tree_options = {.enable_lru_eviction = false,
                                                              .eviction_threshold = 0.9f,
                                                              .enable_path_compression = false,
                                                              .min_compression_length = 2,
                                                              .transformer_blocks_num = 2},
                                       .allocator_options = {// Normal things.
                                                             .per_k_token_ele = 1024,
                                                             .per_v_token_ele = 1024,
                                                             .k_dtype = mllm::kFloat16,
                                                             .v_dtype = mllm::kFloat16,

                                                             // CUDA things.
                                                             .enable_cuda = false,
                                                             .cuda_mem_base = 0x100000,

                                                             // CPU things.
                                                             .enable_cpu_hierarchy_memory = true,
                                                             .zen_fs_options = {
                                                                 .record = false,
                                                                 .working_dir = ".",
                                                                 .blob_bits_size = 20,
                                                                 .page_bits = 7,
                                                                 .lane_bits = 5,
                                                                 .per_k_token_ele = 1024,
                                                                 .per_v_token_ele = 1024,
                                                                 .k_dtype = mllm::kFloat16,
                                                                 .v_dtype = mllm::kFloat16,
                                                                 .mmap_type = mllm::prefix_cache::ZenFSBlobMMapType::kAnonymous,
                                                             }}};

  // TEST 1: alloc 8192 tokens to force ZenFS use mmap
  {
    mllm::prefix_cache::Cache cache(opt);
    std::vector<mllm::prefix_cache::vp_addr_t> addrs;
    addrs.reserve(8192);
    for (int i = 0; i < 8192; i++) { addrs.push_back(cache.alloc(mllm::kCPU)); }
  }
})
