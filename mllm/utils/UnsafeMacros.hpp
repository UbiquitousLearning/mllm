// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define __MLLM_UNSAFE_COMPILER_GCC_OR_CLANG 1
#elif defined(_MSC_VER)
#define __MLLM_UNSAFE_COMPILER_MSVC 1
#endif

#if defined(__MLLM_UNSAFE_COMPILER_GCC_OR_CLANG)
#define __MLLM_UNSAFE_OPT_BEGIN_O3 _Pragma("GCC push_options") _Pragma("GCC optimize(\"O3\")")
#define __MLLM_UNSAFE_OPT_BEGIN_FAST_MATH _Pragma("GCC push_options") _Pragma("GCC optimize(\"fast-math\")")
#define __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH \
  _Pragma("GCC push_options") _Pragma("GCC optimize(\"O3\")") _Pragma("GCC optimize(\"fast-math\")")
#define __MLLM_UNSAFE_OPT_END _Pragma("GCC pop_options")

#elif defined(__MLLM_UNSAFE_COMPILER_MSVC)
#define __MLLM_UNSAFE_OPT_BEGIN_O3 \
  __pragma(optimize("", push)) __pragma(optimize("t", on)) __pragma(optimize("y", on)) __pragma(optimize("g", on))

#define __MLLM_UNSAFE_OPT_BEGIN_FAST_MATH \
  __pragma(optimize("", push)) __pragma(float_control(except, off, push)) __pragma(fp_contract(on))

#define __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH                                                                       \
  __pragma(optimize("", push)) __pragma(optimize("t", on)) __pragma(optimize("y", on)) __pragma(optimize("g", on)) \
      __pragma(float_control(except, off, push)) __pragma(fp_contract(on))

#define __MLLM_UNSAFE_OPT_END __pragma(optimize("", pop)) __pragma(float_control(except, pop))

#else
#define __MLLM_UNSAFE_OPT_BEGIN_O3
#define __MLLM_UNSAFE_OPT_BEGIN_FAST_MATH
#define __MLLM_UNSAFE_OPT_BEGIN_O3_FAST_MATH
#define __MLLM_UNSAFE_OPT_END
#pragma message("Warning: Unknown compiler - optimization macros will have no effect")
#endif
