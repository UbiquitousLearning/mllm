//==============================================================================
//
// Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//--------------------------------------------------------------------------------
/// @file
///   This file includes APIs for the command line parsing on supported platforms
//--------------------------------------------------------------------------------

#pragma once

namespace pal {
// we implement a similar API for POSIX.2
// so that some global var are necessary

extern const char *g_optArg;
extern int g_optInd;

enum {
  no_argument       = 0,
  required_argument = 1,
  optional_argument = 2,
};

//--------------------------------------------------------------------------------------------------
/// @brief
///   This structure describes a single long option name for the sake of getopt_long. The argument
///   longopts must be an array of these structures, one for each long option. Terminate the array
///   with an element containing all zeros.
//--------------------------------------------------------------------------------------------------
struct Option {
  //--------------------------------------------------------------------------------------------------
  /// @brief The name of the long option.
  //--------------------------------------------------------------------------------------------------
  const char *name;

  //--------------------------------------------------------------------------------------------------
  /// @brief
  ///   If the option does not take an argument, no_argument (or 0).
  ///   If the option requires an argument, required_argument (or 1).
  //--------------------------------------------------------------------------------------------------
  int hasArg;

  //--------------------------------------------------------------------------------------------------
  /// @brief
  ///   Specifies how results are returned for a long option.
  ///   If flag is NULL, then GetOptLongOnly() returns val. Otherwise, it returns 0, and flag
  ///   points to a variable which is set to val if the option is found, but
  ///   left unchanged if the option is not found.
  //--------------------------------------------------------------------------------------------------
  int *flag;

  //--------------------------------------------------------------------------------------------------
  /// @brief
  ///   The value to return, or to load into the variable pointed to by flag.
  ///   The last element of the array has to be filled with zeros.
  //--------------------------------------------------------------------------------------------------
  int val;
};

//--------------------------------------------------------------------------------------------------
/// @brief
///   This parses command-line options as POSIX getopt_long_only()
///   but we don't support optstring and optonal_argument now
/// @param argc
///   Argument count
/// @param argv
///   Argument array
/// @param optstring
///   Legitimate option characters, short options, don't support now
/// @param longopts
///   A pointer to the first element of an array of struct option,
///   has_arg field in the struct option indicates 3 possibilities,
///   no_argument, required_argument or optional_argument. we don't
///   support optional_argument now
/// @param longindex
///   If longindex is not NULL, it points to a variable which is set
///   to the index of the long option relative to longopts
/// @return
///   -1 for parsing done, '?' for non-recognized arguments, 0 for
///   flag in longopts is not NULL and saved the val to it
//--------------------------------------------------------------------------------------------------
int getOptLongOnly(int argc,
                   const char *const argv[],
                   const char *optstring,
                   const struct Option *longopts,
                   int *longindex);

}  // namespace pal
