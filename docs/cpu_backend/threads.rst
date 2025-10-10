Parallel API and Thread Configuration in MLLM
==================================================

Introduction
------------

MLLM provides a flexible parallel computing framework that allows CPU kernels to utilize multiple threads for improved performance. The parallel execution is controlled through the Parallel API defined in ``mllm/core/Parallel.hpp`` and configured via CMake build options.

Parallel API
------------

The Parallel API provides macros for parallel execution that abstract away the underlying threading implementation. The API automatically selects the appropriate threading backend based on build configuration:

1. **Apple Grand Central Dispatch (GCD)** - Used on Apple platforms when ``MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD`` is enabled
2. **OpenMP** - Used on most platforms when ``MLLM_KERNEL_THREADS_VENDOR_OPENMP`` is enabled
3. **Sequential execution** - Used when threading is disabled

Key Parallel API Macros
~~~~~~~~~~~~~~~~~~~~~~~

- ``MLLM_AUTO_PARALLEL_BEGIN(__iter__, __num__)`` / ``MLLM_AUTO_PARALLEL_END()`` - Execute a loop with ``__num__`` iterations in parallel
- ``MLLM_AUTO_PARALLEL_FOR_BEGIN(__iter__, __start__, __end__, __step__)`` / ``MLLM_AUTO_PARALLEL_FOR_END()`` - Execute a for-loop in parallel
- ``MLLM_SET_NUM_THREADS(num_threads)`` - Set the number of threads to use for parallel execution
- ``MLLM_SERIAL_FOR_BEGIN(__iter__, __start__, __end__, __step__)`` / ``MLLM_SERIAL_FOR_END()`` - Execute a serial for-loop
- ``MLLM_CONDITIONAL_PARALLEL_FOR(condition, num_threads, iter, start, end, step, ...)`` - Conditionally execute a loop in parallel or serial

MLLM_CONDITIONAL_PARALLEL_FOR
-----------------------------

The ``MLLM_CONDITIONAL_PARALLEL_FOR`` is a macro that provides conditional parallel execution based on a specified condition. It allows switching between parallel and serial execution modes.

Syntax
~~~~~~

.. code-block:: cpp

   MLLM_CONDITIONAL_PARALLEL_FOR(condition, num_threads, iter, start, end, step, body)

Parameters
~~~~~~~~~~

- ``condition``: A boolean expression that determines whether to execute in parallel or serial mode. If true, parallel execution is used; if false, serial execution is used.
- ``num_threads``: The number of threads to use in parallel mode.
- ``iter``: The loop iterator variable name.
- ``start``: The starting value of the iterator.
- ``end``: The ending value of the iterator (exclusive).
- ``step``: The increment step for each iteration.
- ``body``: The loop body code to execute in each iteration.

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The macro is defined as follows:

.. code-block:: cpp

   #define MLLM_CONDITIONAL_PARALLEL_FOR(condition, num_threads, iter, start, end, step, ...) \
     do { \
       if (condition) { \
         MLLM_SET_NUM_THREADS(num_threads); \
         MLLM_AUTO_PARALLEL_FOR_BEGIN(iter, start, end, step){__VA_ARGS__} MLLM_AUTO_PARALLEL_FOR_END() \
       } else { \
         MLLM_SET_NUM_THREADS(1); \
         MLLM_SERIAL_FOR_BEGIN(iter, start, end, step){__VA_ARGS__} MLLM_SERIAL_FOR_END() \
       } \
     } while (0)

In the current implementation:
- ``MLLM_SET_NUM_THREADS`` is a no-op macro that doesn't actually set thread count
- ``MLLM_AUTO_PARALLEL_FOR_BEGIN`` and ``MLLM_SERIAL_FOR_BEGIN`` both expand to simple for-loops
- The key difference is in the intent - one path is meant for parallel execution and the other for serial

Usage Example
~~~~~~~~~~~~~~

.. code-block:: cpp

   const bool use_parallel = options_.getThreads() > 1;
   const int thread_count = options_.getThreads();

   MLLM_CONDITIONAL_PARALLEL_FOR(use_parallel, thread_count, i, 0, N, 1, {
       // Loop body code
       process_element(i);
   });

In this example:
- If ``use_parallel`` is true (when ``options_.getThreads() > 1``), the loop will execute with the specified number of threads
- If ``use_parallel`` is false, the loop will execute serially with a single thread

Capture Mechanism
~~~~~~~~~~~~~~~~~~

Since ``MLLM_CONDITIONAL_PARALLEL_FOR`` expands to standard for-loops, variable capture follows the standard C++ rules:

1. **Direct Variable Access**: Variables from the enclosing scope are directly accessible within the loop body
2. **Non-const Access**: Variables can be modified within the loop body (subject to their own const-ness)
3. **No Special Capture Syntax**: Unlike lambdas or blocks, there's no explicit capture clause - all visible variables are accessible

This is different from lambda expressions or GCD blocks where explicit capture mechanisms are required. The macro simply generates regular for-loops, so standard C++ scoping and access rules apply.

Usage in CPU Kernels
--------------------

CPU kernels use the Parallel API to parallelize operations across data elements. Here's an example of how it's used in the gelu activation function:

.. code-block:: cpp

   if (thread_cnt > 1) {
     MLLM_SET_NUM_THREADS(thread_cnt);
     int tails = N % 4;
     int loops = N - tails;
     MLLM_AUTO_PARALLEL_FOR_BEGIN(i, 0, loops, 4) {
       // Process 4 elements at a time in parallel
       float32x4_t x = vld1q_f32(X + i);
       // ... vectorized computations ...
       vst1q_f32(Z + i, result);
     }
     MLLM_AUTO_PARALLEL_FOR_END()
     // Handle remaining elements serially
     for (; i < N; i++) {
       // ... scalar computations ...
     }
   } else {
     // Serial execution
     // ... regular loop implementation ...
   }

In this example:

1. If ``thread_cnt > 1``, the kernel uses parallel execution
2. ``MLLM_SET_NUM_THREADS`` sets the desired number of threads
3. ``MLLM_AUTO_PARALLEL_FOR_BEGIN`` and ``MLLM_AUTO_PARALLEL_FOR_END`` define the parallel loop section
4. Vectorized operations are performed on chunks of data (4 elements at a time for float32)
5. Remaining elements that don't fit in chunks are handled serially

Another example from cast_types.cpp shows how to use the parallel macros with conditional handling:

.. code-block:: cpp

   if (thread_count > 1) {
     MLLM_SET_NUM_THREADS(thread_count);
     MLLM_AUTO_PARALLEL_FOR_BEGIN(i, 0, len, 4)
     int remain = len - i;
     if (remain >= 4) {
       int32x4_t v32_src = vld1q_s32(src + i);
       vst1q_f32(dst + i, vcvtq_f32_s32(v32_src));
     } else {
       for (int j = i; j < len; j++) { dst[j] = (mllm_fp32_t)src[j]; }
     }
     MLLM_AUTO_PARALLEL_FOR_END();
   } else {
     // Serial implementation
   }

CMake Thread Configuration
--------------------------

MLLM provides several CMake options to configure threading support:

Threading Options
~~~~~~~~~~~~~~~~~

- ``MLLM_KERNEL_USE_THREADS`` (default: ON) - Enable or disable threading support entirely
- ``MLLM_KERNEL_THREADS_VENDOR_OPENMP`` (default: ON) - Enable OpenMP threading
- ``MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD`` (default: OFF) - Enable Apple Grand Central Dispatch threading

Platform-Specific Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apple Platforms
^^^^^^^^^^^^^^^

On Apple platforms (macOS, iOS), MLLM supports both OpenMP and GCD threading models:

.. code-block:: shell
   :caption: Example CMake configuration for Apple platforms

   -DMLLM_KERNEL_USE_THREADS=ON
   -DMLLM_KERNEL_THREADS_VENDOR_OPENMP=ON
   -DMLLM_KERNEL_THREADS_VENDOR_APPLE_GCD=OFF

If both OpenMP and GCD are enabled, GCD takes precedence with a warning message.

Non-Apple Platforms
^^^^^^^^^^^^^^^^^^^

On non-Apple platforms, OpenMP is typically used:

.. code-block:: shell
   :caption: Example CMake configuration for non-Apple platforms

   -DMLLM_KERNEL_USE_THREADS=ON
   -DMLLM_KERNEL_THREADS_VENDOR_OPENMP=ON

Best Practices
--------------

1. **Conditional Parallelization**: Only use parallel execution when there's enough work to justify the overhead:

   .. code-block:: cpp
   
      if (thread_count > 1 && len > 1024 * 4) {
        // Parallel implementation
      } else {
        // Serial implementation
      }

2. **Proper Chunking**: Divide work into appropriately sized chunks for better load balancing:

   .. code-block:: cpp
   
      size_t chunk_size = (vec_size + thread_count - 1) / thread_count;
      chunk_size = (chunk_size + lanes - 1) & ~(lanes - 1);

3. **Handling Remainders**: Always handle data that doesn't fit evenly into vectorized chunks:

   .. code-block:: cpp
   
      // Process main chunks in parallel
      MLLM_AUTO_PARALLEL_FOR_BEGIN(i, 0, vec_size, lanes) {
        // Vectorized operations
      }
      MLLM_AUTO_PARALLEL_FOR_END()
      
      // Handle remainder elements serially
      if (vec_size < size) {
        // Process remaining elements
      }

Conclusion
----------

The Parallel API in MLLM provides a flexible and portable way to parallelize CPU kernel operations. Through CMake configuration options, developers can choose the appropriate threading backend for their platform while the API abstracts away the implementation details. CPU kernels can leverage these macros to achieve better performance on multi-core systems while maintaining code clarity and portability.
