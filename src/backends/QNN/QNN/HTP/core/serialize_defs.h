//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SERIALIZE_DEFS_H
#define SERIALIZE_DEFS_H 1

#include <cstdint>
#include <typeinfo>

// General class of op: foreground for main thread, vec for HVX thread(s), mtx
// for HMX thread(s).
enum OpStoreType {
    OpStoreFg,
    OpStoreVec,
    OpStoreMtx,
};

class Op;

namespace hnnx {

class Serializer;
class Deserializer;

/**
 * @brief Common base to register error in Serialize/Deserialize
 * Calling register_error sets the error string (unless there is one already).
 * any_error() should be checked after each full use of serialize/deserialize
 * Note: register_error must be called with a string constant, or other
 * persistent string, since the contents of the string are not copied.
 *
 */
class DeSerError {
    char const *errstr = nullptr; // null if no error
  public:
    void reset_error() { errstr = nullptr; }
    bool any_error() const { return errstr != nullptr; }
    void register_error(char const *estr)
    { // must be a persistent string!
        if (errstr == nullptr) errstr = estr;
    }
    char const *error_string() const { return errstr; }
};

// We allow 4 bits of extra flag storage when storing an Op type, using
// the upper four bits of the index.
enum {
    SerializeOpFlagMask = 0xf0000000,
    SerializeOpFlagShift = 28,
};
void op_serialize_common(Serializer &sctx, Op const *op, std::type_info const *actual_type = nullptr);

static constexpr unsigned OP_SEQNO_MARKER_XOR = 0x1303ee71u;
static constexpr unsigned OP_SEQNO_MARKER_MASK = 0x1FFFFFFFu; // upper 3 bits reserved for flags.
static constexpr unsigned OP_SEQNO_PRELOAD_FLAG = 0x80000000u;

} // namespace hnnx

#endif // SERIALIZE_DEFS_H
