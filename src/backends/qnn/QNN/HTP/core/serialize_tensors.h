//==============================================================================
//
// Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SERIALIZE_TENSORS_H
#define SERIALIZE_TENSORS_H 1

#include <cstdio>
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <string_view>
#include "limits.h"
#include "log.h"

#include "forward_classes.h"
#include "serdes_tensors.h"
#include "minihash.h"

namespace hnnx {

// see comment in serdes_tensors_common.h for an overview of how this works.

class SerTensorConn : public SerTensorConnDefs {
    tensor_idx previous_index = 0;
    // map Tensor* to the id assigned to it (either a sequential index, or a temporary forward index)
    minihash_noerase<ptr_type, tensor_idx> tensor_index_map;
    std::vector<int> forward_allocation_table;
    int forward_allocation_free = -1;
    int fwd_pending_count = 0; // for stats
    int resolved_pending_count = 0;

    // a list of 'pending' update pairs {seq_index,fwd_index}
    std::vector<std::pair<tensor_idx, tensor_idx>> pending_update_pairs;

  public:
    SerTensorConn() {}
    void tensor_def(Serializer &, ptr_type);
    void tensor_ref(Serializer &, ptr_type);
    void tensor_refs(Serializer &, ptr_type const *, unsigned);
    void store_pending(Serializer &);

  protected:
    void flush_pending(Serializer &);
};

} // namespace hnnx

#endif
