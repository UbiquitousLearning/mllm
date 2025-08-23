// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#ifdef MLLM_KERNEL_USE_THREADS
//===----------------------------------------------------------------------===//
// APPLE! ONLY Apple can do! Do not use openmp on iOS/OSX.
//===----------------------------------------------------------------------===//
#if defined(__APPLE__) && defined(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD)
#include <dispatch/dispatch.h>
#include <cstddef>
// Parallel primitives
#define MLLM_AUTO_PARALLEL_BEGIN(__iter__, __num__) \
dispatch_apply(__num__, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t __iter__) {
#define MLLM_AUTO_PARALLEL_END() \
  });

#define MLLM_AUTO_PARALLEL_FOR_BEGIN(__iter__, __start__, __end__, __step__)            \
  {                                                                                     \
    size_t __start_val__ = (__start__);                                                 \
    size_t __end_val__ = (__end__);                                                     \
    size_t __step_val__ = (__step__);                                                   \
    size_t __count__ = (__end_val__ - __start_val__ + __step_val__ - 1) / __step_val__; \
    dispatch_apply(__count__, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t __idx__) { \
        size_t __iter__ = __start_val__ + __idx__ * __step_val__;
#define MLLM_AUTO_PARALLEL_FOR_END() \
  });                                \
  }

#define MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(__iter__, __start__, __end__, __step__, __num_threads__) \
  {                                                                                              \
    size_t __start_val__ = (__start__);                                                          \
    size_t __end_val__ = (__end__);                                                              \
    size_t __step_val__ = (__step__);                                                            \
    size_t __count__ = (__end_val__ - __start_val__ + __step_val__ - 1) / __step_val__;          \
    dispatch_apply(__count__, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t __idx__) { \
        size_t __iter__ = __start_val__ + __idx__ * __step_val__;
#define MLLM_AUTO_PARALLEL_FOR_END_NT() \
  });                                   \
  }

// Apple GCD does not need explicit thread count setting
#define MLLM_SET_NUM_THREADS(num_threads) \
  do { (void)(num_threads); } while (0)
#endif  // defined(__APPLE__) && defined(MLLM_KERNEL_THREADS_VENDOR_APPLE_GCD)

//===----------------------------------------------------------------------===//
// OpenMP.
//===----------------------------------------------------------------------===//
#if defined(MLLM_KERNEL_THREADS_VENDOR_OPENMP)
// #include <omp.h>

#define OMP_PRAGMA(x) _Pragma(#x)

#define MLLM_AUTO_PARALLEL_BEGIN(__iter__, __num__) \
  OMP_PRAGMA(omp parallel for)                      \
  for (int __iter__ = 0; __iter__ < __num__; ++__iter__) {
#define MLLM_AUTO_PARALLEL_END() }

#define MLLM_AUTO_PARALLEL_FOR_BEGIN(__iter__, __start__, __end__, __step__) \
  OMP_PRAGMA(omp parallel for)                                               \
  for (long long __iter__ = (__start__); __iter__ < (__end__); __iter__ += (__step__)) {
#define MLLM_AUTO_PARALLEL_FOR_END() }

#define MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(__iter__, __start__, __end__, __step__, __num_threads__) \
  OMP_PRAGMA(omp parallel for num_threads(__num_threads__))                                      \
  for (long long __iter__ = (__start__); __iter__ < (__end__); __iter__ += (__step__)) {
#define MLLM_AUTO_PARALLEL_FOR_END_NT() }

// #define MLLM_SET_NUM_THREADS(num_threads) omp_set_num_threads(num_threads)

#define MLLM_SET_NUM_THREADS(num_threads) \
  do { (void)(num_threads); } while (0)

#endif  // defined(MLLM_KERNEL_THREADS_VENDOR_OPENMP)
#else   // MLLM_KERNEL_USE_THREADS

#define MLLM_AUTO_PARALLEL_BEGIN(__iter__, __num__) for (int __iter__ = 0; __iter__ < __num__; ++__iter__) {
#define MLLM_AUTO_PARALLEL_END() }

#define MLLM_AUTO_PARALLEL_FOR_BEGIN(__iter__, __start__, __end__, __step__) \
  for (long long __iter__ = (__start__); __iter__ < (__end__); __iter__ += (__step__)) {
#define MLLM_AUTO_PARALLEL_FOR_END() }

#define MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(__iter__, __start__, __end__, __step__, __num_threads__) \
  for (long long __iter__ = (__start__); __iter__ < (__end__); __iter__ += (__step__)) {
#define MLLM_AUTO_PARALLEL_FOR_END_NT() }

#define MLLM_SET_NUM_THREADS(num_threads) \
  do { (void)(num_threads); } while (0)

#endif  // MLLM_KERNEL_USE_THREADS

#define MLLM_SERIAL_FOR_BEGIN(__iter__, __start__, __end__, __step__) \
  for (long long __iter__ = (__start__); __iter__ < (__end__); __iter__ += (__step__)) {
#define MLLM_SERIAL_FOR_END() }

#define MLLM_CONDITIONAL_PARALLEL_FOR(condition, num_threads, iter, start, end, step, ...)                              \
  do {                                                                                                                  \
    if (condition) {                                                                                                    \
      MLLM_AUTO_PARALLEL_FOR_BEGIN_NT(iter, start, end, step, num_threads){__VA_ARGS__} MLLM_AUTO_PARALLEL_FOR_END_NT() \
    } else {                                                                                                            \
      MLLM_SERIAL_FOR_BEGIN(iter, start, end, step){__VA_ARGS__} MLLM_SERIAL_FOR_END()                                  \
    }                                                                                                                   \
  } while (0)
