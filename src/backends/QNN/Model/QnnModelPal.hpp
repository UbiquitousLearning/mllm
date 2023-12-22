//==============================================================================
//
//  Copyright (c) 2021-2022 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

namespace qnn_wrapper_api {

// specify this address to distingiush from NULL pointer
#define DL_DEFAULT (void *)(0x4)

//---------------------------------------------------------------------------
/// @brief
///   obtain address of a symbol in a shared object or executable
/// @handle
///   a handle of a dynamic loaded shared object returned by dlopen
/// @symbol
///   a null-terminated symbol name
/// @return
///   On success, return the address associated with symbol
///   On error, NULL
//---------------------------------------------------------------------------
void *dlSym(void *handle, const char *symbol);

//---------------------------------------------------------------------------
/// @brief
///   obtain error diagnostic for functions in the dl-family APIs.
/// @return
///   returns a human-readable, null-terminated string describing the most
///   recent error that occurred from a call to one of the functions in the
///   dl-family APIs.
///
//---------------------------------------------------------------------------
char *dlError(void);

//---------------------------------------------------------------------------
/// @brief
///   Returns a pointer to a null-terminated byte string, which contains copies
///   of at most maxlen bytes from the string pointed to by str. If the null
///   terminator is not encountered in the first maxlen bytes, it is added to
///   the duplicated string.
/// @source
///   Null-terminated source string.
/// @maxlen
///   Max number of bytes to copy from str
/// @return
///   A pointer to the newly allocated string, or a null pointer if an error
///   occurred.
///
//---------------------------------------------------------------------------
char *strnDup(const char *source, size_t maxlen);
}  // namespace qnn_wrapper_api