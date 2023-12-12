//============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//============================================================================

#pragma once

#define DEBUG_ON 0

#if DEBUG_ON
#define DEBUG_MSG(...)            \
  {                               \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n");        \
  }
#else
#define DEBUG_MSG(...)
#endif
