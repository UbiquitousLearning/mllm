================
MLLM LM Cache
================

The MLLM LM Cache module provides an efficient Key-Value caching mechanism for optimizing inference performance of large language models and multimodal models. This module supports both static and dynamic caching strategies, effectively reducing redundant computations and improving inference speed.

Overview
========

In Transformer architecture models, the attention mechanism needs to maintain key-value caches to avoid recomputing representations of historical tokens. MLLM provides multiple cache implementations to meet different performance and memory requirements:

- **StaticCache**: Pre-allocates fixed-size cache, suitable for scenarios with known maximum sequence length
- **DynamicCache**: Dynamically allocates cache, suitable for variable-length sequence scenarios
- **SubStaticCache**: A sub-view of static cache, supporting cache slicing operations

API Reference
=============

StaticCache
-----------

Pre-allocates fixed-size cache, suitable for performance optimization during inference.

.. code-block:: cpp

   #include "mllm/nn/lmcache/StaticCache.hpp"

   // Create static cache
   auto cache = mllm::nn::StaticCache(
       max_cache_length,  // Maximum cache length
       layer_nums,        // Number of layers
       q_heads,          // Number of query heads
       kv_heads,         // Number of key-value heads
       kv_dims,          // Key-value dimensions
       k_dtype,          // Key data type
       v_dtype,          // Value data type
       device_type,      // Device type (kCPU, kOpenCL, etc.)
       use_fa2           // Whether to use FlashAttention2
   );

   // Update cache
   auto [k_cached, v_cached] = cache.updateKVCache(layer_idx, k_tensor, v_tensor);

   // Get current sequence length
   int32_t seq_len = cache.getCurrentSeqCnt(layer_idx);

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------+----------------+-------------------------------+
| Parameter         | Type           | Description                   |
+===================+================+===============================+
| max_cache_length  | int32_t        | Maximum cache sequence length |
+-------------------+----------------+-------------------------------+
| layer_nums        | int32_t        | Number of model layers        |
+-------------------+----------------+-------------------------------+
| q_heads           | int32_t        | Number of query attention     |
|                   |                | heads                         |
+-------------------+----------------+-------------------------------+
| kv_heads          | int32_t        | Number of key-value attention |
|                   |                | heads                         |
+-------------------+----------------+-------------------------------+
| kv_dims           | int32_t        | Key-value dimensions          |
+-------------------+----------------+-------------------------------+
| k_dtype           | DataTypes      | Key tensor data type          |
+-------------------+----------------+-------------------------------+
| v_dtype           | DataTypes      | Value tensor data type        |
+-------------------+----------------+-------------------------------+
| device_type       | DeviceTypes    | Device type (default kCPU)    |
+-------------------+----------------+-------------------------------+
| use_fa2           | bool           | Whether to use FlashAttention2|
|                   |                | (default true)                |
+-------------------+----------------+-------------------------------+

DynamicCache
------------

Dynamically allocates cache, suitable for training or variable-length inference scenarios.

.. code-block:: cpp

   #include "mllm/nn/lmcache/DynamicCache.hpp"

   // Create dynamic cache
   auto cache = mllm::nn::DynamicCache(
       layer_nums,  // Number of layers
       q_heads,     // Number of query heads
       kv_heads,    // Number of key-value heads
       kv_dims,     // Key-value dimensions
       use_fa2      // Whether to use FlashAttention2
   );

   // Update cache
   auto [k_cached, v_cached] = cache.updateKVCache(layer_idx, k_tensor, v_tensor);

   // Get current sequence length
   int32_t seq_len = cache.getCurrentSeqCnt();

SubStaticCache
--------------

A sub-view of static cache that allows slicing operations on the cache.

.. code-block:: cpp

   // Create sub-cache from existing static cache
   auto sub_cache = mllm::nn::SubStaticCache(
       parent_cache,  // Parent cache reference
       start_idx,     // Start index
       len           // Length
   );

   // Use in the same way as StaticCache
   auto [k_cached, v_cached] = sub_cache.updateKVCache(layer_idx, k_tensor, v_tensor);

Tensor Format
=============

Non-FlashAttention2 Mode
------------------------

Input tensor format: ``[Batch, Heads, Sequence, Dimension]``

.. code-block:: cpp

   // Example: single batch, 32 heads, sequence length 1, dimension 128
   Tensor k = Tensor::random({1, 32, 1, 128});
   Tensor v = Tensor::random({1, 32, 1, 128});

FlashAttention2 Mode
--------------------

Input tensor format: ``[Batch, Sequence, Heads, Dimension]``

.. code-block:: cpp

   // Example: single batch, sequence length 1, 32 heads, dimension 128
   Tensor k = Tensor::random({1, 1, 32, 128});
   Tensor v = Tensor::random({1, 1, 32, 128});

Usage Examples
==============

Basic Usage
-----------

.. code-block:: cpp

   #include "mllm/nn/lmcache/StaticCache.hpp"

   // Configure parameters
   const int32_t max_seq_len = 2048;
   const int32_t num_layers = 24;
   const int32_t num_q_heads = 32;
   const int32_t num_kv_heads = 8;  // Support GQA (Grouped Query Attention)
   const int32_t head_dim = 128;

   // Create cache
   auto cache = mllm::nn::StaticCache(
       max_seq_len, num_layers, num_q_heads, num_kv_heads, head_dim,
       mllm::DataTypes::kFP16, mllm::DataTypes::kFP16, mllm::DeviceTypes::kCPU
   );

   // Use in inference loop
   for (int layer = 0; layer < num_layers; ++layer) {
       // Assume k, v are key-value tensors of current layer
       auto [k_cache, v_cache] = cache.updateKVCache(layer, k, v);
       
       // Use cached key-values for attention computation
       auto attention_output = attention_func(q, k_cache, v_cache);
   }

Dynamic Cache Example
---------------------

.. code-block:: cpp

   #include "mllm/nn/lmcache/DynamicCache.hpp"

   auto dynamic_cache = mllm::nn::DynamicCache(num_layers, num_q_heads, num_kv_heads, head_dim);

   // Build cache step by step
   for (int step = 0; step < max_steps; ++step) {
       for (int layer = 0; layer < num_layers; ++layer) {
           auto [k_cache, v_cache] = dynamic_cache.updateKVCache(layer, k_step, v_step);
           // Process current step
       }
   }

Performance Optimization
=============================

Memory Layout Optimization
---------------------------

- **CPU**: Uses ``memcpy`` for efficient memory copying
- **GPU/NPU**: Uses tensor's ``copy2`` method for device-optimized copying operations

GQA Support
-----------

Supports Grouped Query Attention by calculating the repeat factor through ``q_heads / kv_heads``, automatically handling cases where the number of key-value heads is less than query heads.

Device-Specific Optimization
----------------------------

.. code-block:: cpp

   // CPU optimization path
   case kCPU: {
       // Use memcpy for block copying
       std::memcpy(cache_ptr, input_ptr, copy_size);
       break;
   }

   // GPU/NPU optimization path
   default: {
       // Use tensor operations for device-optimized copying
       input_tensor.copy2(cache_tensor);
       break;
   }

Important Notes
===============

1. **Memory Pre-allocation**: StaticCache pre-allocates all memory during construction, suitable for scenarios with known maximum sequence length
2. **FA2 Compatibility**: Different attention implementations require different tensor layouts, ensure to choose the correct ``use_fa2`` parameter
3. **Device Compatibility**: Ensure cache and input tensors are on the same device
4. **Data Types**: Supports mixed precision, keys and values can use different data types

Error Handling
==============

.. code-block:: cpp

   // Check sequence length limits
   if (current_seq_len + input_seq_len > max_cache_length) {
       throw std::runtime_error("Sequence length exceeds cache capacity");
   }

   // Validate tensor shapes
   MLLM_RT_ASSERT_EQ(k.shape()[1], kv_heads);
   MLLM_RT_ASSERT_EQ(v.shape()[1], kv_heads);

Best Practices
==============

1. **Choose Appropriate Cache Type**:
   
   - Use ``StaticCache`` for inference to achieve optimal performance
   - Use ``DynamicCache`` for training or variable-length scenarios

2. **Memory Management**:
   
   - Estimate maximum sequence length to avoid memory shortage
   - Consider using ``SubStaticCache`` for memory slicing

3. **Performance Tuning**:
   
   - Choose appropriate data types based on hardware characteristics
   - Enable FlashAttention2 for better memory efficiency

Related Documentation
=====================

- `MLLM Architecture Documentation <../arch/index.rst>`_
- `CPU Backend Optimization <../cpu_backend/index.rst>`_
- `API Reference <../api/index.rst>`_
