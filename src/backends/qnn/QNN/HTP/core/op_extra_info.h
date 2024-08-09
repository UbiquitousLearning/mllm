//==============================================================================
//
// Copyright (c) 2020-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef OP_EXTRA_INFO_H
#define OP_EXTRA_INFO_H 1

#include <utility>

#include "interface_defs.h"

namespace hnnx {

// map Op* to a few properties, to avoid the need to keep them in the Op
// object. Currently contains the ID and the gate/done checkpoint indices.
struct OpExtraInfo {
    using Chkpts = std::pair<int, int>;

    OpId id;
    Chkpts chkpts;
    const char *op_tag;
    explicit OpExtraInfo(OpId id_in) : id(id_in), chkpts(-1, -1) {}
    OpExtraInfo(OpId id_in, int cg, int dc) : id(id_in), chkpts(cg, dc) {}
    OpExtraInfo() : OpExtraInfo(0) {}

    bool valid() const { return id != 0; };
    void clear() { id = 0; };
};

} // namespace hnnx

#endif // OP_EXTRA_INFO_H
