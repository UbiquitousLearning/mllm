//==============================================================================
//
// Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

#include "QnnModelPal.hpp"

namespace qnn_wrapper_api {
void *dlSym(void *handle, const char *symbol) {
  if (handle == DL_DEFAULT) {
    return ::dlsym(RTLD_DEFAULT, symbol);
  }

  return ::dlsym(handle, symbol);
}

char *dlError(void) { return ::dlerror(); }

char *strnDup(const char *source, size_t maxlen) { return ::strndup(source, maxlen); }
}  // namespace qnn_wrapper_api