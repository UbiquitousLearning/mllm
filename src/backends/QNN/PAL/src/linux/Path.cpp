//==============================================================================
//
//  Copyright (c) 2008-2014, 2015, 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <stdlib.h>

#include <sstream>
#ifndef PATH_MAX
#include <limits.h>
#endif

#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

char pal::Path::getSeparator() { return '/'; }

std::string pal::Path::combine(const std::string &s1, const std::string &s2) {
  std::stringstream ss;
  ss << s1;
  if (s1.size() > 0 && s1[s1.size() - 1] != getSeparator()) {
    ss << getSeparator();
  }
  ss << s2;
  return ss.str();
}

std::string pal::Path::getDirectoryName(const std::string &path) {
  std::string rc = path;
  size_t index   = path.find_last_of(pal::Path::getSeparator());
  if (index != std::string::npos) {
    rc = path.substr(0, index);
  }
  return rc;
}

std::string pal::Path::getAbsolute(const std::string &path) {
  // Functionality was duplicated of function in FileOp
  // Just call that function directly instead
  return pal::FileOp::getAbsolutePath(path);
}

bool pal::Path::isAbsolute(const std::string &path) {
  return path.size() > 0 && path[0] == getSeparator();
}
