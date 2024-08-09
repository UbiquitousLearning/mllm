//==============================================================================
//
//  Copyright (c) 2008-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#ifndef __QNXNTO__
#include <sys/sendfile.h>
#endif
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
#ifdef __QNXNTO__
static bool is_qnx_dir(const struct dirent *ep) {
  struct dirent_extra *exp;
  bool is_dir = false;

  for (exp = _DEXTRA_FIRST(ep); _DEXTRA_VALID(exp, ep); exp = _DEXTRA_NEXT(exp)) {
    if (exp->d_type == _DTYPE_STAT || exp->d_type == _DTYPE_LSTAT) {
      struct stat *statbuff = &((dirent_extra_stat *)exp)->d_stat;
      if (statbuff && S_ISDIR(statbuff->st_mode)) {
        is_dir = true;
        break;
      }
    }
  }
  return is_dir;
}
#endif

// ------------------------------------------------------------------------------
//    pal::Directory::create
// ------------------------------------------------------------------------------
bool pal::Directory::create(const std::string &path, pal::Directory::DirMode dirmode) {
  struct stat st;
  int status = 0;
  if (stat(path.c_str(), &st) != 0) {
    // Directory does not exist
    status = mkdir(path.c_str(), static_cast<mode_t>(dirmode));
  } else if (!S_ISDIR(st.st_mode)) {
    errno  = ENOTDIR;
    status = -1;
  }
  return (status == 0);
}

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
bool pal::Directory::remove(const std::string &dirName) {
  DIR *dir;
  struct dirent *entry;

  dir = opendir(dirName.c_str());
  if (dir == nullptr) {
    // If the directory doesn't exist then just return true.
    if (errno == ENOENT) {
      return true;
    }
    return false;
  }

#ifdef __QNXNTO__
  if (dircntl(dir, D_SETFLAG, D_FLAG_STAT) == -1) {
    return false;
  }
#endif

  // Recursively traverse the directory tree.
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
      std::stringstream ss;
      ss << dirName << Path::getSeparator() << entry->d_name;
      std::string path = ss.str();
#ifdef __QNXNTO__
      if (is_qnx_dir(entry))
#else
      if (entry->d_type == DT_DIR)
#endif
      {
        // It's a directory so we need to drill down into it and delete
        // its contents.
        if (!remove(path)) {
          return false;
        }
      } else {
        if (::remove(path.c_str())) {
          return false;
        }
      }
    }
  }

  closedir(dir);

  if (::remove(dirName.c_str())) {
    return false;
  }

  return true;
}

bool pal::Directory::makePath(const std::string &path) {
  struct stat st;
  bool rc = false;

  if (path == ".") {
    rc = true;
  } else if (stat(path.c_str(), &st) == 0) {
    if (st.st_mode & S_IFDIR) {
      rc = true;
    }
  } else {
    size_t offset = path.find_last_of(Path::getSeparator());
    if (offset != std::string::npos) {
      std::string newPath = path.substr(0, offset);
      if (!makePath(newPath)) {
        return false;
      }
    }

    // There is a possible race condition, where a file/directory can be
    // created in between the stat() above, and the mkdir() call here.
    // So, ignore the return code from the mkdir() call, and then re-check
    // for existence of the directory after it. Ensure both that it exists
    // and that it is a directory - just like above.
    mkdir(path.c_str(), 0777);

    if ((stat(path.c_str(), &st) == 0) && (st.st_mode & S_IFDIR)) {
      rc = true;
    }
  }

  return rc;
}
