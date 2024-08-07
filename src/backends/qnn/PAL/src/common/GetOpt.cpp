//=============================================================================
//
//  Copyright (c) 2020-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <string.h>

#include <string>

#include "PAL/GetOpt.hpp"

using namespace std;

namespace pal {

const char *g_optArg = nullptr;
int g_optInd         = 1;

static const struct Option *findOpt(const string str,
                                    const struct Option *longopts,
                                    int *longindex) {
  const struct Option *opt = nullptr;
  int idx                  = 0;
  size_t searchEnd         = str.find_first_of("=");

  for (opt = longopts; opt->name && strlen(opt->name) > 0; opt++, idx++) {
    if (str.substr(0, searchEnd) == opt->name) {
      if (longindex) {
        *longindex = idx;
      }
      break;
    }
  }
  // if not found, opt would point to the last element of longopts
  // whose name MUST be empty
  return opt->name ? opt : nullptr;
}

int getOptLongOnly(int argc,
                   const char *const argv[],
                   const char *,
                   const struct Option *longopts,
                   int *longindex) {
  const struct Option *opt;
  int argLen      = 0;
  bool isShort    = false;
  const char *arg = "";

  g_optArg = nullptr;
  // no arg, means the end of command
  if (g_optInd >= argc) {
    return -1;
  }

  arg = argv[g_optInd];

  if (arg[0] != '-') {
    g_optInd += 1;
    return '?';
  }

  argLen = strlen(arg);

  if (argLen < 2) {
    g_optInd += 1;
    return '?';
  }

  if (!longopts) {
    g_optInd += 1;
    return '?';
  }

  // check short options with this form, -a arg
  if (argLen == 2) {
    isShort = true;
    // check short options with this form, -a=arg
  } else if (argLen > 3 && arg[2] == '=') {
    isShort = true;
    // check for long options, can be used for both forms
  } else if (argLen > 2 && arg[1] != '=') {
    if (arg[1] != '-') {
      g_optInd += 1;
      return '?';
    }
    isShort = false;
  }

  // start after -- to find the option
  const char *const optStr = isShort ? &arg[1] : &arg[2];
  opt                      = findOpt(optStr, longopts, longindex);
  if (!opt) {
    g_optInd += 1;
    return '?';
  }

  if (opt->hasArg == no_argument) {
    g_optInd += 1;

    if (!opt->flag) {
      return opt->val;
    } else {
      *(opt->flag) = opt->val;
      return 0;
    }
  }

  if (opt->hasArg == required_argument) {
    string optStr    = argv[g_optInd];
    size_t assignIdx = optStr.find_first_of("=");
    bool advance     = (assignIdx == string::npos);

    // if it is --opt arg form, this will be true,
    // so we need to advance one step to get arg
    // otherwise, need to stop advance step & extract arg from argv[g_optInd]
    if (advance) {
      g_optInd += 1;
    }

    if (g_optInd >= argc) {
      return '?';
    } else {
      // if advance, means it is the form --opt arg
      // otherwise, the form, --opt=arg
      if (advance) {
        // since g_optInd is advanced, g_optArg can be assigned directly
        g_optArg = argv[g_optInd];
      } else {
        if (assignIdx == optStr.size()) {
          return '?';
        }
        // for not advanced form,
        // g_optArg should point to the address right after "="
        g_optArg = &argv[g_optInd][assignIdx + 1];
      }
      // OK, now we are ready to handle the next pair
      g_optInd += 1;

      if (!opt->flag) {
        return opt->val;
      } else {
        *(opt->flag) = opt->val;
        return 0;
      }
    }
  }

  return '?';
}  // end of getOptLongOnly

}  // namespace pal
