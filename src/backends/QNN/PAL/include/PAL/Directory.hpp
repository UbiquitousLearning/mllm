//==============================================================================
//
//  Copyright (c) 2008-2014, 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//---------------------------------------------------------------------------
/// @file
///   This file includes APIs for directory operations on supported platforms
//---------------------------------------------------------------------------

#pragma once

#include <string>

#include "PAL/FileOp.hpp"

namespace pal {
class Directory;
}

class pal::Directory {
 public:
  using DirMode = pal::FileOp::FileMode;
  //---------------------------------------------------------------------------
  /// @brief
  ///   Creates a directory in the file system.
  /// @param path
  ///   Name of directory to create.
  /// @param dirmode
  ///   Directory mode
  /// @return
  ///   True if
  ///     1. create a directory successfully
  ///     2. or directory exist already
  ///   False otherwise
  ///
  ///  For example:
  ///
  ///  - Create a directory in default.
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///     pal::Directory::Create(path, pal::Directory::DirMode::S_DEFAULT_);
  ///     pal::Directory::Create(path);
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///
  ///  - Create a directory with specific permission.
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///     pal::Directory::Create(path, pal::Directory::DirMode::S_IRWXU_|
  ///                                  pal::Directory::DirMode::S_IRWXG_|
  ///                                  pal::Directory::DirMode::S_IRWXO_);
  ///     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  ///
  /// @note For windows, dirmode is not used.
  /// @note For linux, dirmode is used to set the permission of the folder.
  //---------------------------------------------------------------------------
  static bool create(const std::string &path,
                     pal::Directory::DirMode dirmode = pal::Directory::DirMode::S_DEFAULT_);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Removes the entire directory whether it's empty or not.
  /// @param path
  ///   Name of directory to delete.
  /// @return
  ///   True if the directory was successfully deleted, false otherwise.
  //---------------------------------------------------------------------------
  static bool remove(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Creates a directory and all parent directories required.
  /// @param path
  ///   Path of directory to create.
  /// @return
  ///   True if the directory was successfully created, false otherwise.
  //---------------------------------------------------------------------------
  static bool makePath(const std::string &path);
};
