//==============================================================================
//
//  Copyright (c) 2008-2014, 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//==============================================================================

//------------------------------------------------------------------------------
/// @file
///   The file includes APIs for path related operations on supported platforms
//------------------------------------------------------------------------------

#pragma once

#include <string>
#include <vector>

namespace pal {
class Path;
}

class pal::Path {
 public:
  //---------------------------------------------------------------------------
  /// @brief Returns path separator for the system
  //---------------------------------------------------------------------------
  static char getSeparator();

  //---------------------------------------------------------------------------
  /// @brief Concatenate s1 and s2
  //---------------------------------------------------------------------------
  static std::string combine(const std::string &s1, const std::string &s2);

  //---------------------------------------------------------------------------
  /// @brief Get the directory name
  //---------------------------------------------------------------------------
  static std::string getDirectoryName(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Get absolute path
  //---------------------------------------------------------------------------
  static std::string getAbsolute(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Check if the input path is absolute path
  //---------------------------------------------------------------------------
  static bool isAbsolute(const std::string &path);

 private:
};
