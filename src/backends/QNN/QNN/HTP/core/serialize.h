//==============================================================================
//
// Copyright (c) 2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SERIALIZE_H
#define SERIALIZE_H 1

// ****************************** NOTE *********************************
// This is for support of external builds which #include "serialize.h".
//
// For new work, #include one or more of the below, instead of "serialize.h":
//
//  serialize_defs.h:
//     -forward decls of Deserializer, Serializer; also declares DeSerError and the SerializeOpFlag values
//  deserializer.h:
//     - declares Deserializer, and related routines needed on decode side. Includes 'serialize_defs.h'
//  seraliazer.h:
//     - declares Serializer, and derived classes (FileSerializer, NullSerializer). Includes 'serialize_defs.h'
//
// Forward decls of Deserializer, Serializer are also in "forward_classes.h".
// ****************************** NOTE *********************************

#include "serializer.h"
#include "deserializer.h"

#endif // SERIALIZE_H
