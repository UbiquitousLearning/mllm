//==============================================================================
//
// Copyright (c) 2018-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//-----------------------------------------------------------------------------
/// @file
///   The file inludes APIs for string operations on supported platforms
//-----------------------------------------------------------------------------

#pragma once

#include <sys/types.h>

namespace pal {
class StringOp;
}

//------------------------------------------------------------------------------
/// @brief
///   FileOp contains OS Specific file system functionality.
//------------------------------------------------------------------------------
class pal::StringOp {
 public:
  //---------------------------------------------------------------------------
  /// @brief
  ///   Copy copy_size bytes from buffer src to buffer dst. Behaviour of the
  ///   function is undefined if src and dst overlap.
  /// @param dst
  ///   Destination buffer
  /// @param dst_size
  ///   Size of destination buffer
  /// @param src
  ///   Source buffer
  /// @param copy_size
  ///   Number of bytes to copy
  /// @return
  ///   Number of bytes copied
  //---------------------------------------------------------------------------
  static size_t memscpy(void *dst, size_t dstSize, const void *src, size_t copySize);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Returns a pointer to a null-terminated byte string, which contains copies
  ///   of at most size bytes from the string pointed to by str. If the null
  ///   terminator is not encountered in the first size bytes, it is added to the
  ///   duplicated string.
  /// @param source
  ///   Source string
  /// @param maxlen
  ///   Max number of bytes to copy from str
  /// @return
  ///   A pointer to the newly allocated string, or a null pointer if an error
  ///   occurred.
  //---------------------------------------------------------------------------
  static char *strndup(const char *source, size_t maxlen);
};
