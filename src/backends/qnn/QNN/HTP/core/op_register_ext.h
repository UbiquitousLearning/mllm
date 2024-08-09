//==============================================================================
//
// Copyright (c) 2021,2023 Qualcomm Technologies, Inc.
// All Rights Reserved.
// Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef OP_REGISTER_EXT_H
#define OP_REGISTER_EXT_H

#include "graph_status.h"
#include "template_help.h"
#include "op_utils.h"
#include "op_info.h"
#include "op_registry.h"
#include "template_help_tensor_ext.h"
#include "serialize_register.h"
#include "op_register_types.h"
#include "macros_attribute.h"
#include "weak_linkage.h"

PUSH_VISIBILITY(default)

//
// Interface for HTP op packages. This is a reduced subset of capability compared to QNN.
//

// Optional termination function.  Perform any shutdown and return success if
// OK.  May be ommitted.
typedef GraphStatus (*PackageOpTermFn_t)();

// Interface class.  An op package is dynamically loaded, then the special
// function op_pkg_init is loaded and called.  It takes a reference argument to
// a PackageOpIf.
//
// In addition to specifying the name and optional termination function, this
// function should perform any relevant op and optimization rule registration.
// It's possible that this function may be called more than once, though we try
// to avoid it.  So, to be on the safe side, it should return immediately with
// GraphStatus::Success if it's already been called.
//
// _name must be non-null and non-empty.  It's used as a unique key into the
// registry, to avoid duplicate loading of op packages, should one be specified
// more than once in the list of options.
//
// _term may be null.
struct PackageOpIf {
    const char *_name = nullptr;
    PackageOpTermFn_t _term = nullptr;
};

// Entry point function for the op package.
typedef GraphStatus (*PackageOpInitFn_t)(PackageOpIf &);

namespace hnnx {

class PackageOpStorageBase {
  public:
    const std::string op_name;
    const std::string_view type_tag;
    const SimpleOpFactory simpop;
    const std::type_info &type_info;
    const Op::tensor_deserializer_register_func deserializer_reg_func;
    const deserialize_op_func deserialize_func;
    cost_function_t cost_f;
    const Flags_word flags;

    API_EXPORT PackageOpStorageBase(const std::string_view op_name_in, const std::string_view type_tag_in,
                                    const SimpleOpFactory simpop_in, const std::type_info &tinf,
                                    const Op::tensor_deserializer_register_func deserializer_reg_func_in,
                                    const deserialize_op_func deserialize_func_in, const cost_function_t cost_f_in,
                                    Flags_word flags_in);

    API_EXPORT const OpFactory make_op_wrapper() const;
};

// The map to store op package ops
API_EXPORT std::map<std::string, std::vector<std::unique_ptr<PackageOpStorageBase>> *> &get_pkg_op_tmp_map();

} // namespace hnnx

POP_VISIBILITY()

#define INIT_PKG_CORE_INIT_FUNC()                                                                                      \
    static bool sg_init = false;                                                                                       \
    extern "C" int op_pkg_init(PackageOpIf &pkg_if)                                                                    \
    {                                                                                                                  \
        pkg_if._name = THIS_PKG_NAME_STR;                                                                              \
        if (sg_init) {                                                                                                 \
            return GraphStatus::Success;                                                                               \
        }                                                                                                              \
        REGISTER_PACKAGE_OPS();                                                                                        \
        REGISTER_PACKAGE_OPTIMIZATIONS()                                                                               \
        sg_init = true;                                                                                                \
        return GraphStatus::Success;                                                                                   \
    }

#define INIT_PACKAGE_OP_DEF()                                                                                          \
    API_HIDDEN std::vector<std::unique_ptr<hnnx::PackageOpStorageBase>> &current_package_ops_storage_vec_func()        \
    {                                                                                                                  \
        static std::vector<std::unique_ptr<hnnx::PackageOpStorageBase>> opv;                                           \
        return opv;                                                                                                    \
    }                                                                                                                  \
    extern "C" {                                                                                                       \
    void clearPackageOpsStorageVecFunc() { current_package_ops_storage_vec_func().clear(); }                           \
    }

#define DECLARE_PACKAGE_OP_DEF()                                                                                       \
    API_HIDDEN std::vector<std::unique_ptr<hnnx::PackageOpStorageBase>> &current_package_ops_storage_vec_func();

#define REGISTER_PACKAGE_OPS()                                                                                         \
    if (hnnx::get_pkg_op_tmp_map().find(std::string(THIS_PKG_NAME_STR)) == hnnx::get_pkg_op_tmp_map().end()) {         \
        hnnx::get_pkg_op_tmp_map()[std::string(THIS_PKG_NAME_STR)] = &current_package_ops_storage_vec_func();          \
        hnnx::pkg_ops_opts_registration();                                                                             \
    }

/** @brief Create an Op type's type suffix from argument types */
#define PKG_TYPE_SUFFIX(OP, ARGS) (hnnx::ConcatStr<hnnx::ConstexprStrLen(OP), hnnx::ConstexprStrLen(ARGS)>(OP, ARGS))

#ifndef OP_REG_COMPILE
#define DEF_PACKAGE_OP(F, OP)                                                                                          \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, Flags::RESOURCE_HVX)                                                                \
    APPEND_REG_OP_ELEM_NO_TCM_FOLDING(                                                                                 \
            F, THIS_PKG_NAME_STR "::" OP,                                                                              \
            PKG_TYPE_SUFFIX(THIS_PKG_NAME_STR "::" OP, hnnx::ArgsTuples2<F>::inputTypeNames), true)
#else
#define DEF_PACKAGE_OP(F, OP) __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#endif

using package_cost_function_t = float (*)(Op const *);
inline float call_cost_func(package_cost_function_t func, const Op *op)
{
    return (func)(op);
}
inline float call_cost_func(std::string_view, const Op *op)
{
    return 0.0;
}
namespace hnnx {
template <auto F>
void add_package_op_ext(std::vector<std::unique_ptr<PackageOpStorageBase>> &ops, const std::string_view op_name_in,
                        const char *type_tag, const package_cost_function_t cost_f_in, Flags_word flags_in);
}

#ifndef OP_REG_COMPILE
#define DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F, OP, COST, ...)                                                            \
    COST_OF(F, COST)                                                                                                   \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, __VA_ARGS__)                                                                        \
    APPEND_REG_OP_ELEM_NO_TCM_FOLDING(                                                                                 \
            F, THIS_PKG_NAME_STR "::" OP,                                                                              \
            PKG_TYPE_SUFFIX(THIS_PKG_NAME_STR "::" OP, hnnx::ArgsTuples2<F>::inputTypeNames), true)

#define DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F, OP, COST_F, ...)                                                        \
    COST_OF_F(F, [](const Graph &, const Op *op) -> float { return call_cost_func(COST_F, op); })                      \
    FLAGS_FOR_DT_NO_TCM_FOLDING(F, __VA_ARGS__)                                                                        \
    APPEND_REG_OP_ELEM_NO_TCM_FOLDING(                                                                                 \
            F, THIS_PKG_NAME_STR "::" OP,                                                                              \
            PKG_TYPE_SUFFIX(THIS_PKG_NAME_STR "::" OP, hnnx::ArgsTuples2<F>::inputTypeNames), true)
#else
#define DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F, OP, COST, ...)     __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#define DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F, OP, COST_F, ...) __reg_op__(F, OP)<<<__FILE__, __LINE__>>>
#endif

DECLARE_PACKAGE_OP_DEF()

#endif // OP_REGISTER_EXT_H
