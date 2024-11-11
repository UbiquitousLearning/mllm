//==============================================================================
//
// Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <dlfcn.h>
#include <stdlib.h>
#include <vector>

#include "Log.h"
#include "PAL/Debug.hpp"
#include "PAL/DynamicLoading.hpp"
const std::vector<std::string> LIB_PREFIX = {"/system/lib64/", "/odm/lib64/", "/vendor/lib64/", "/data/local/tmp/mllm/qnn-lib/", "/system_ext/lib64/"};
void *pal::dynamicloading::dlOpen(const char *filename, int flags) {
  int realFlags = 0;

  if (flags & DL_NOW) {
    realFlags |= RTLD_NOW;
  }

  if (flags & DL_LOCAL) {
    realFlags |= RTLD_LOCAL;
  }

  if (flags & DL_GLOBAL) {
    realFlags |= RTLD_GLOBAL;
  }

  auto res = ::dlopen(filename, realFlags);
  if (!res) {
      for (auto prefix_ : LIB_PREFIX) {
          std::string prefix = prefix_ + filename;
          res = ::dlopen(prefix.c_str(), realFlags);
          if (res) {
              break;
          }
          MLLM_LOG_ERROR("{} not found", prefix);
      }
  }
  return res;
}

void *pal::dynamicloading::dlSym(void *handle, const char *symbol) {
  if (handle == DL_DEFAULT) {
      handle = RTLD_DEFAULT;
  }

  return ::dlsym(handle, symbol);
}

int pal::dynamicloading::dlAddrToLibName(void *addr, std::string &name) {
  // Clean the output buffer
  name = std::string();

  // If the address is empty, return zero as treating failure
  if (!addr) {
    DEBUG_MSG("Input address is nullptr.");
    return 0;
  }

  // Dl_info do not maintain the lifetime of its string members,
  // it would be maintained by dlopen() and dlclose(),
  // so we do not need to release it manually
  Dl_info info;
  int result = ::dladdr(addr, &info);

  // If dladdr() successes, set name to the library name
  if (result) {
    name = std::string(info.dli_fname);
  } else {
    DEBUG_MSG("Input address could not be matched to a shared object.");
  }

  return result;
}

int pal::dynamicloading::dlClose(void *handle) {
  if (!handle) {
    return 0;
  }

  return ::dlclose(handle);
}

char *pal::dynamicloading::dlError(void) { return ::dlerror(); }
