//==============================================================================
//
//  Copyright (c) 2008-2013,2015,2019-2022 Qualcomm Technologies, Inc.
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
#include <limits.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "PAL/Debug.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"

typedef struct stat Stat_t;

//---------------------------------------------------------------------------
//    pal::FileOp::HasFileExtension
//---------------------------------------------------------------------------
bool pal::FileOp::checkFileExists(const std::string& fileName) {
  Stat_t sb;

  if (stat(fileName.c_str(), &sb) == -1) {
    return false;
  } else {
    return true;
  }
}

//---------------------------------------------------------------------------
//    pal::FileOp::move
//---------------------------------------------------------------------------
bool pal::FileOp::move(const std::string& currentName, const std::string& newName, bool overwrite) {
  if (overwrite) {
    remove(newName.c_str());
  }
  return (rename(currentName.c_str(), newName.c_str()) == 0);
}

//---------------------------------------------------------------------------
//    pal::FileOp::deleteFile
//---------------------------------------------------------------------------
bool pal::FileOp::deleteFile(const std::string& fileName) {
  return (remove(fileName.c_str()) == 0);
}

//------------------------------------------------------------------------------
// pal::FileOp::checkIsDir
//------------------------------------------------------------------------------
bool pal::FileOp::checkIsDir(const std::string& fileName) {
  bool retVal = false;
  Stat_t sb;
  if (stat(fileName.c_str(), &sb) == 0) {
    if (sb.st_mode & S_IFDIR) {
      retVal = true;
    }
  }
  return retVal;
}

//------------------------------------------------------------------------------
//    pal::FileOp::getFileInfo
//------------------------------------------------------------------------------
bool pal::FileOp::getFileInfo(const std::string& filename,
                              pal::FileOp::FilenamePartsType_t& filenameParts) {
  std::string name;

  // Clear the result
  filenameParts.basename.clear();
  filenameParts.extension.clear();
  filenameParts.directory.clear();

  size_t lastPathSeparator = filename.find_last_of(Path::getSeparator());
  if (lastPathSeparator == std::string::npos) {
    // No directory
    name = filename;
  } else {
    // has a directory part
    filenameParts.directory = filename.substr(0, lastPathSeparator);
    name                    = filename.substr(lastPathSeparator + 1);
  }

  size_t ext = name.find_last_of(".");
  if (ext == std::string::npos) {
    // no extension
    filenameParts.basename = name;
  } else {
    // has extension
    filenameParts.basename  = name.substr(0, ext);
    filenameParts.extension = name.substr(ext + 1);
  }

  return true;
}

//---------------------------------------------------------------------------
//    pal::FileOp::copyOverFile
//---------------------------------------------------------------------------
bool pal::FileOp::copyOverFile(const std::string& fromFile, const std::string& toFile) {
  bool rc = false;
  int readFd;
  int writeFd;
  struct stat statBuf;

  // Open the input file.
  readFd = ::open(fromFile.c_str(), O_RDONLY);
  if (readFd == -1) {
    close(readFd);
    return false;
  }

  // Stat the input file to obtain its size. */
  if (fstat(readFd, &statBuf) != 0) {
    close(readFd);
    return false;
  }

  // Open the output file for writing, with the same permissions as the input
  writeFd = ::open(toFile.c_str(), O_WRONLY | O_CREAT | O_TRUNC, statBuf.st_mode);
  if (writeFd == -1) {
    close(readFd);
    return false;
  }

  // Copy the file in a non-kernel specific way */
  char fileBuf[8192];
  ssize_t rBytes, wBytes;
  while (true) {
    rBytes = read(readFd, fileBuf, sizeof(fileBuf));

    if (!rBytes) {
      rc = true;
      break;
    }

    if (rBytes < 0) {
      rc = false;
      break;
    }

    wBytes = write(writeFd, fileBuf, (size_t)rBytes);

    if (!wBytes) {
      rc = true;
      break;
    }

    if (wBytes < 0) {
      rc = false;
      break;
    }
  }

  /* Close up. */
  close(readFd);
  close(writeFd);
  return rc;
}

static bool getFileInfoListRecursiveImpl(const std::string& path,
                                         pal::FileOp::FilenamePartsListType_t& filenamePartsList,
                                         const bool ignoreDirs,
                                         size_t maxDepth) {
  struct dirent** namelist = nullptr;
  int entryCount           = 0;

  // Base case
  if (maxDepth == 0) {
    return true;
  }

#ifdef __ANDROID__
  // android dirent.h has the wrong signature for alphasort so it had to be disabled or fixed
  entryCount = scandir(path.c_str(), &namelist, 0, 0);
#else
  entryCount = scandir(path.c_str(), &namelist, 0, alphasort);
#endif
  if (entryCount < 0) {
    return false;
  } else {
    while (entryCount--) {
      const std::string dName(namelist[entryCount]->d_name);
      free(namelist[entryCount]);

      // skip current directory, prev directory and empty string
      if (dName.empty() || dName == "." || dName == "..") {
        continue;
      }

      std::string curPath = path;
      curPath += pal::Path::getSeparator();
      curPath += dName;

      // recurse if directory but avoid symbolic links to directories
      if (pal::FileOp::checkIsDir(curPath)) {
        Stat_t sb;
        if (lstat(curPath.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
          if (!getFileInfoListRecursiveImpl(curPath, filenamePartsList, ignoreDirs, maxDepth - 1)) {
            return false;
          }
        }

        if (ignoreDirs) {
          continue;
        }

        // Append training / to make this path look like a directory for
        // getFileInfo()
        if (curPath.back() != pal::Path::getSeparator()) {
          curPath += pal::Path::getSeparator();
        }
      }

      // add to vector
      pal::FileOp::FilenamePartsType_t filenameParts;
      if (pal::FileOp::getFileInfo(curPath, filenameParts)) {
        filenamePartsList.push_back(filenameParts);
      }
    }

    free(namelist);
  }

  return true;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileInfoList
//---------------------------------------------------------------------------
bool pal::FileOp::getFileInfoList(const std::string& path,
                                  FilenamePartsListType_t& filenamePartsList) {
  return getFileInfoListRecursiveImpl(path, filenamePartsList, false, 1);
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileInfoListRecursive
//---------------------------------------------------------------------------
bool pal::FileOp::getFileInfoListRecursive(const std::string& path,
                                           FilenamePartsListType_t& filenamePartsList,
                                           const bool ignoreDirs) {
  return getFileInfoListRecursiveImpl(
      path, filenamePartsList, ignoreDirs, std::numeric_limits<size_t>::max());
}

//---------------------------------------------------------------------------
//    pal::FileOp::getAbsolutePath
//---------------------------------------------------------------------------
std::string pal::FileOp::getAbsolutePath(const std::string& path) {
  // NOTE: This implementation is broken currently when a path with
  // non-existant components is passed! NEO-19723 was created to address.
  char absPath[PATH_MAX + 1] = {0};

  if (realpath(path.c_str(), absPath) == NULL) {
    DEBUG_MSG("GetAbsolute path fail! Error code : %d", errno);
    return std::string();
  }
  return std::string(absPath);
}

//---------------------------------------------------------------------------
//    pal::FileOp::setCWD
//---------------------------------------------------------------------------
bool pal::FileOp::setCurrentWorkingDirectory(const std::string& workingDir) {
  return chdir(workingDir.c_str()) == 0;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getDirectory
//---------------------------------------------------------------------------
std::string pal::FileOp::getDirectory(const std::string& file) {
  std::string rc = file;
  size_t offset  = file.find_last_of(Path::getSeparator());
  if (offset != std::string::npos) {
    rc = file.substr(0, offset);
  }
  return rc;
}

//---------------------------------------------------------------------------
//    pal::FileOp::getFileName
//---------------------------------------------------------------------------
std::string pal::FileOp::getFileName(const std::string& file) {
  std::string rc = file;
  size_t offset  = file.find_last_of(Path::getSeparator());
  if (offset != std::string::npos) {
    rc = file.substr(offset + 1);  // +1 to skip path separator
  }
  return rc;
}

//---------------------------------------------------------------------------
//    pal::FileOp::hasFileExtension
//---------------------------------------------------------------------------
bool pal::FileOp::hasFileExtension(const std::string& file) {
  FilenamePartsType_t parts;
  getFileInfo(file, parts);

  return !parts.extension.empty();
}

//---------------------------------------------------------------------------
//    pal::FileOp::getCWD
//---------------------------------------------------------------------------
std::string pal::FileOp::getCurrentWorkingDirectory() {
  char buffer[PATH_MAX + 1];
  buffer[0] = '\0';

  // If there is any failure return empty string. It is technically possible
  // to handle paths exceeding PATH_MAX on some flavors of *nix but platforms
  // like Android (Bionic) do no provide such capability. For consistency we
  // will not handle extra long path names.
  if (nullptr == getcwd(buffer, PATH_MAX)) {
    return std::string();
  } else {
    return std::string(buffer);
  }
}

//---------------------------------------------------------------------------
//    pal::FileOp::partsToString
//---------------------------------------------------------------------------
std::string pal::FileOp::partsToString(const FilenamePartsType_t& filenameParts) {
  std::string path;

  if (!filenameParts.directory.empty()) {
    path += filenameParts.directory;
    path += Path::getSeparator();
  }
  if (!filenameParts.basename.empty()) {
    path += filenameParts.basename;
  }
  if (!filenameParts.extension.empty()) {
    path += ".";
    path += filenameParts.extension;
  }
  return path;
}
