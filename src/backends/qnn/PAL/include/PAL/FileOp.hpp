//==============================================================================
//
// Copyright (c) 2008-2022 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

//------------------------------------------------------------------------------
/// @file
///   This file includes APIs for file operations on the supported platforms
//------------------------------------------------------------------------------

#pragma once

#include <fcntl.h>

#include <string>
#include <vector>

namespace pal {
class FileOp;
}

//------------------------------------------------------------------------------
/// @brief
///   FileOp contains OS Specific file system functionality.
//------------------------------------------------------------------------------
class pal::FileOp {
 public:
  // enum for symbolic constants mode, strictly follow linux usage
  // windows or another OS user should transfer the usage
  // ref : http://man7.org/linux/man-pages/man2/open.2.html
  enum class FileMode : uint32_t {
    S_DEFAULT_ = 0777,
    S_IRWXU_   = 0700,
    S_IRUSR_   = 0400,
    S_IWUSR_   = 0200,
    S_IXUSR_   = 0100,
    S_IRWXG_   = 0070,
    S_IRGRP_   = 0040,
    S_IWGRP_   = 0020,
    S_IXGRP_   = 0010,
    S_IRWXO_   = 0007,
    S_IROTH_   = 0004,
    S_IWOTH_   = 0002,
    S_IXOTH_   = 0001
  };

  //---------------------------------------------------------------------------
  /// @brief
  ///   Copies a file from one location to another, overwrites if the
  ///   destination already exists.
  /// @param source
  ///   File name of the source file.
  /// @param target
  ///   File name of the target file.
  /// @return
  ///   True on success, otherwise false.
  //---------------------------------------------------------------------------
  static bool copyOverFile(const std::string &source, const std::string &target);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Checks whether the file exists or not.
  /// @param fileName
  ///   File name of the source file, including its complete path.
  /// @return
  ///   True on success, otherwise false.
  //---------------------------------------------------------------------------
  static bool checkFileExists(const std::string &fileName);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Renames an existing file. If the file with target name exists, this call
  ///   overwrites it with the file with source name.
  /// @param source
  ///   Current File name.
  /// @param target
  ///   New name of the file.
  /// @param overwrite
  ///   Flag indicating to overwrite existing file with newName
  /// @return
  ///   True if successful, otherwise false.
  /// @warning
  ///   Does not work if source and target are on different filesystems.
  //---------------------------------------------------------------------------
  static bool move(const std::string &source, const std::string &target, bool overwrite);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Delete an existing file
  /// @param fileName
  ///   File name of the file to be deleted.
  /// @return
  ///   True if successful, otherwise false.
  //---------------------------------------------------------------------------
  static bool deleteFile(const std::string &fileName);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Check if path is a directory or not
  /// @param path
  ///   Path to check
  /// @return
  ///   True if successful, otherwise false.
  //---------------------------------------------------------------------------
  static bool checkIsDir(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Data type representing parts of a filename
  //---------------------------------------------------------------------------
  typedef struct {
    //---------------------------------------------------------------------------
    /// @brief Name of the file without the extension (i.e., basename)
    //---------------------------------------------------------------------------
    std::string basename;

    //---------------------------------------------------------------------------
    /// @brief Name of the file extension (i.e., .txt or .hlnd, .html)
    //---------------------------------------------------------------------------
    std::string extension;

    //---------------------------------------------------------------------------
    /// @brief
    ///   Location of the file (i.e., /abc/xyz/foo.bar <-- /abc/xyz/).
    ///   If the file name has no location then the Directory points to
    ///   empty string
    //---------------------------------------------------------------------------
    std::string directory;
  } FilenamePartsType_t;

  //---------------------------------------------------------------------------
  /// @brief
  ///   Determines the components of a given filename, being the directory,
  ///   basename and extension. If the file has no location or extension, these
  ///   components remain empty
  /// @param filename
  ///   Path of the file for which the components are to be determined
  /// @param filenameParts
  ///   Will contain the file name components when this function returns
  /// @return
  ///   True if successful, false otherwise
  //---------------------------------------------------------------------------
  static bool getFileInfo(const std::string &filename, FilenamePartsType_t &filenameParts);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Typedef for a vector of FilenamePartsType_t
  //---------------------------------------------------------------------------
  typedef std::vector<FilenamePartsType_t> FilenamePartsListType_t;

  //---------------------------------------------------------------------------
  /// @brief
  ///   Typedef for a vector of FilenamePartsType_t const iterator
  //---------------------------------------------------------------------------
  typedef std::vector<FilenamePartsType_t>::const_iterator FilenamePartsListTypeIter_t;

  //---------------------------------------------------------------------------
  /// @brief
  ///   Returns a vector of FilenamePartsType_t objects for a given directory
  /// @param path
  ///   Path to scan for files
  /// @return
  ///   True if successful, false otherwise
  //---------------------------------------------------------------------------
  static bool getFileInfoList(const std::string &path, FilenamePartsListType_t &filenamePartsList);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Returns a vector of FilenamePartsType_t objects for a given directory
  ///   and the child directories inside.
  /// @param path
  ///   Path to directory to scan for files for
  ///   @note if path is not a directory - the function will return false
  /// @param filenamePartList
  ///   List to append to
  /// @param ignoreDirs
  ///   If this flag is set to true, directories (and symbolic links to directories)
  ///   are not included in the list. Only actual files below the specified
  ///   directory path will be appended.
  /// @return True if successful, false otherwise
  /// @note Directories in list only populate Directory member variable of the struct.
  ///       That is Basename and Extension will be empty strings.
  /// @note Symbolic links to directories are not followed. This is to avoid possible
  ///       infinite recursion. However the initial call to this method can have
  ///       path to be a symbolic link to a directory. If ignoreDirs is true,
  ///       symbolic links to directories are also ignored.
  /// @note The order in which the files/directories are listed is platform
  ///       dependent. However files inside a directory always come before the
  ///       directory itself.
  //---------------------------------------------------------------------------
  static bool getFileInfoListRecursive(const std::string &path,
                                       FilenamePartsListType_t &filenamePartsList,
                                       const bool ignoreDirs);

  //---------------------------------------------------------------------------
  /// @brief
  ///   Create an absolute path from the supplied path
  /// @param path
  ///   Path should not contain trailing '/' or '\\'
  /// @return
  ///   Return absolute path without trailing '/' or '\\'
  //---------------------------------------------------------------------------
  static std::string getAbsolutePath(const std::string &path);

  //---------------------------------------------------------------------------
  /// @brief Get the file name from a path
  //---------------------------------------------------------------------------
  static std::string getFileName(const std::string &file);

  //---------------------------------------------------------------------------
  /// @brief Get the directory path to a file
  //---------------------------------------------------------------------------
  static std::string getDirectory(const std::string &file);

  //---------------------------------------------------------------------------
  /// @brief Get the current working directory.
  /// @returns The absolute CWD or empty string if the path could not be
  ///          retrieved (because it was too long or deleted for example).
  //---------------------------------------------------------------------------
  static std::string getCurrentWorkingDirectory();

  //---------------------------------------------------------------------------
  /// @brief Set the current working directory
  //---------------------------------------------------------------------------
  static bool setCurrentWorkingDirectory(const std::string &workingDir);

  //---------------------------------------------------------------------------
  /// @brief Returns true if the file contains any extension or false.
  //---------------------------------------------------------------------------
  static bool hasFileExtension(const std::string &file);

  //---------------------------------------------------------------------------
  /// @brief Returns full path of file, Directory/Basename(.Extension, if any)
  //---------------------------------------------------------------------------
  static std::string partsToString(const FilenamePartsType_t &filenameParts);
};
