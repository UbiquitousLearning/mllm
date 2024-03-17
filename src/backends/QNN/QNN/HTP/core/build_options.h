//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef BUILD_OPTIONS_H
#define BUILD_OPTIONS_H 1

namespace build_options {
#ifdef WITH_OPT_DEBUG
constexpr bool WithDebugOpt = true;
#else
constexpr bool WithDebugOpt = false;
#endif
#ifdef DEBUG_TILING
constexpr bool DebugTiling = true;
#else
constexpr bool DebugTiling = false;
#endif

#ifdef PREPARE_DISABLED
static constexpr bool WITH_PREPARE = false;
#else
static constexpr bool WITH_PREPARE = true;
#endif

#ifdef DEBUG_REGISTRY
constexpr bool DebugRegistry = true;
#else
constexpr bool DebugRegistry = false;
#endif

#ifdef DEBUG_TCM
constexpr bool DebugTcm = true;
#else
constexpr bool DebugTcm = false;
#endif

#ifdef __hexagon__
constexpr bool IsPlatformHexagon = true;
#else
constexpr bool IsPlatformHexagon = false;
#endif

#ifdef _WIN32
constexpr bool PLATFORM_CANNOT_BYPASS_VCALL = true;
constexpr bool OS_WIN = true;
#else
constexpr bool PLATFORM_CANNOT_BYPASS_VCALL = false;
constexpr bool OS_WIN = false;
#endif
} // namespace build_options

#endif // BUILD_OPTIONS_H
