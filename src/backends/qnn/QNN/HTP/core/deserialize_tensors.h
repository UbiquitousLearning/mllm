//==============================================================================
//
// Copyright (c) 2021-2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef DESERIALIZE_TENSORS_H
#define DESERIALIZE_TENSORS_H 1

#include <cstdio>
#include <cassert>
#include <cstring>
#include <memory>
#include <vector>
#include <tuple>
#include "limits.h"
#include "log.h"

#include "forward_classes.h"
#include "serdes_tensors.h"

namespace hnnx {

// see comment in serdes_tensors.h for overview of how this works.

class Deserializer;

class DeserTensorConn : public SerTensorConnDefs {
    typedef unsigned tensor_idx;
    typedef Tensor const *ptr_type;

    tensor_idx previous_index = 0;
    // this collects all of the tensor_def we have seen. index is seq_index-1.
    std::vector<ptr_type> defined_tensors;
    // record of tensors ptrs we need to update after their definitions.
    // each time we decode a tensor-ref as FWD_INDEX_FLAG+k, we store the address of the
    // tensor pointer here at [k].
    // (if there already is one in the first slot, we store in the second slot, to
    // support the case where two pointers are being set).
    std::vector<std::pair<ptr_type *, ptr_type *>> pending_tensor_updates;

  public:
    DeserTensorConn() {}
    // process a tensor definition
    void tensor_def(Deserializer &, ptr_type);
    // process a tensor ref
    void tensor_ref(Deserializer &, ptr_type &);
    // process n tensor refs.
    void tensor_refs(Deserializer &, ptr_type *ptrs, unsigned num);

    // read an identity (for use in subsequent need_fixup)
    tensor_idx read_identity(Deserializer &);
    // apply the identity to 'fix' a tensor pointer (usually now, sometimes later
    void need_fixup(tensor_idx ident, ptr_type *dst);

    // done at a point corresponding to SerTensorConn::store_pending()
    void read_pending(Deserializer &);

    inline bool has_pending_updates() { return (pending_tensor_updates.size() > 0); }

    inline void reserve_tensors(const size_t n) { defined_tensors.reserve(n); }

  protected:
    // insert a new update request into the structure.
    void insert_fwd_rec(tensor_idx idx, ptr_type *destp);
    // update the fwd index fwdidx with 'value'; remove it from the forward table.
    bool update_fwd(tensor_idx fwdidx, ptr_type value);
    // this is called when we've read a tensor index, and it looks like an update record;
    // this will process all update records and return the index we wanted, which followed
    // them all.
    tensor_idx process_fwd_pending(Deserializer &, tensor_idx);

    tensor_idx read_identity_inline(Deserializer &);
    void apply_fixup_inline(tensor_idx idx, ptr_type *dst);
};

} // namespace hnnx

#endif // DESERIALIZE_TENSORS_H
