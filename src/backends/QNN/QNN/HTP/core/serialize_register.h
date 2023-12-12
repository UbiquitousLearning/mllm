//==============================================================================
//
// Copyright (c) 2020 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef SERIALIZE_REGISTER_H
#define SERIALIZE_REGISTER_H 1

#include "crate.h"
#include "op_info.h"
#include "macros_attribute.h"
#include "weak_linkage.h"

namespace hnnx {

class SimpleOpWrapper;

template <typename T> struct deserialize_tensor_using_constructor {
    static uptr_Tensor deserialize(const Op *producer, Deserializer &dctx)
    {
        // If the graph has a crate, put the deserialized
        // Tensor into the crate, using a 'Tensor_Deleter' which won't actually try to delete it.
        Crate *const cp = graph_crate(dctx.graph());
        if (cp == nullptr) {
            std::unique_ptr<Tensor> optr = std::make_unique<T>(producer, dctx);
            return optr;
        } else {
            Tensor *const op_ptr = cp->emplace<T>(producer, dctx);
            return std::unique_ptr<Tensor, Tensor_Deleter>(op_ptr, Tensor_Deleter(true));
        }
    }
};

// Allocation/deallocation for Op

template <typename T> struct alloc_func_for_op {
    static void *alloc_func(void *ptr, Deserializer &dctx) { return new (ptr) T(dctx); }
};

API_EXPORT void deserialize_simple_op_wrapper(void *, Deserializer &dctx, std::unique_ptr<SimpleOpBase> sop_in);

template <typename T> struct alloc_func_for_op_ext {
    static void *alloc_func(void *ptr, Deserializer &dctx)
    {
        auto sop = std::make_unique<T>();
        deserialize_simple_op_wrapper(ptr, dctx, std::move(sop));
        return ptr;
    }
};

template <typename T> struct dealloc_func_for_op {
    static std::enable_if_t<(!std::is_trivially_destructible<T>::value)> dealloc_func(Graph *graph_in, void *ptr)
    {
        if constexpr (has_clear<T>) {
            static_cast<T *>(ptr)->clear(graph_in);
        }
        static_cast<T *>(ptr)->~T();
    }
};

template <typename OPTYPE> inline void register_framework_op(char const *opname)
{
    register_op_info(typeid(OPTYPE), hnnx::cost_function_t(StandardCosts::FAST), 0, (SimpleOpFactory) nullptr, false,
                     opname);
    op_deserializer_fn const fn(alloc_func_for_op<OPTYPE>::alloc_func, dealloc_func_for_op<OPTYPE>::dealloc_func,
                                sizeof(OPTYPE), alignof(OPTYPE));
    deserialize_op_register(&typeid(OPTYPE), opname, fn);
}

} // namespace hnnx
#endif // SERIALIZE_REGISTER_H
