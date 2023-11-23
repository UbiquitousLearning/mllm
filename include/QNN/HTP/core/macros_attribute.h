//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef MACROS_MSVC_CPP17_H
#define MACROS_MSVC_CPP17_H

// Macros to compatible with MSVC and clang
#if defined(_MSC_VER)

// Macros to tells the compiler to never inline a particular member function
#define NOINLINE __declspec(noinline)
// Macros to force a function to be inlined, meaning that the function call is replaced with the function body at compile time
#define ALWAYSINLINE __forceinline

#ifndef API_EXPORT
//Macros to export symbol from a library
#define API_EXPORT __declspec(dllexport)
#endif
#ifndef API_HIDDEN
//Macros to hidden symbol from a library
#define API_HIDDEN

//Macros to tell the compiler that the function returns an object that is not aliased, that is, referenced by any other pointers.
#define RESTRICT_VAR __restrict

#endif
#else

#define NOINLINE     __attribute__((noinline))
#define ALWAYSINLINE __attribute((always_inline))

#ifndef API_EXPORT
#define API_EXPORT [[gnu::visibility("default")]]
#endif
#ifndef API_HIDDEN
#define API_HIDDEN [[gnu::visibility("hidden")]]
#endif

#define RESTRICT_VAR __restrict__

#endif // _MSC_VER

#endif //MACROS_MSVC_H
