//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

#define MASK_INFINITY 1e15

BEGIN_PKG_OP_DEFINITION(PKG_CausalMask);

// op execute function declarations
template <typename TensorType>
GraphStatus causalmaskImpl(TensorType &out_0,
                           const TensorType &in_0);

// forward declaration of sample cost function
static float causalmaskCostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((causalmaskImpl<Tensor>), "CausalMask")
 */
DEF_PACKAGE_OP((causalmaskImpl<Tensor>), "CausalMask")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((causalmaskImpl<PlainFloatTensor>), "CausalMask", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((causalmaskImpl<PlainFloatTensor>),
 * "CausalMask", causalmaskCostFunc, Flags::RESOURCE_HVX)
 */

/*
 * optimization definitions
 * need to be global in the package
 * one definition per optimization
 * syntax: DEF_PACKAGE_OPTIMIZATION(PRIORITY,MATCHCODE,CONSTRAINTCODE,REPLACECODE)
 * PRIORITY predefined values include EARLY(2000), MIDDLE(3000), LATE(4000)
 * HTP core provides some replacement functions for op package to use
 * for more information about optimization rules, please refer to HTP core documentations
 */

/*
 * op parameter order definitions
 * need to be global in the package
 * one definition per op, and this is optional
 * syntax: DEF_PACKAGE_PARAM_ORDER(OP,PARAM1,MANDATORY1,DEFAULT1,PARAM2,MANDATORY2,DEFAULT2...)
 * one or more parameters can be specified for each op
 * order of parameters listed determines the order of parameters passed into op execution functions
 * if an op does not have a parameter order definition, parameter order passed into Qnn_addNode
 *   will be passed into op execution functions
 * if an op has a parameter order definition, any parameter passed into Qnn_addNode with unlisted
 *   name will be abandoned
 * if two or more op packages with the same package name will be registered, they cannot list
 *   conflicting parameter orders
 * PARAM refers to parameter name as a string literal
 * MANDATORY refers to whether this parameter is required to be provided at Qnn_addNode
 * DEFAULT is used when MANDATORY is false
 *     if provided as Qnn_Param_t*,
 *       DEFAULT will be used for graph construction when this parameter is not provided at
 *       Qnn_addNode
 *     if provided as nullptr,
 *       graph construction will skip this parameter when this parameter is not provided at
 *       Qnn_addNode
 */

/* execute functions for ops */

template <typename TensorType>
GraphStatus causalmaskImpl(TensorType &out_0,
                           const TensorType &in_0)

{
    /*
     * add code here
     * */
    /*
     * To have good performance and stability, it is required to avoid heap memory
     * allocation in this function. The heap memory allocation includes but not
     * limited to calling malloc, operator new, constructing STL container objects
     * like std::vector with default allocator, and adding items like calling
     * std::vector::push_back to STL container objects with default allocator.
     *
     * Please check in SDK documentation for more information.
     */
    out_0.set_dims(in_0);

    int old_dim = 0;

    // NHSD
    auto [b_in, h_in, w_in, d_in] = in_0.dims();

    // S > 1 => mask
    if (w_in > 1) {
        for (Idx b = 0; b < b_in; b++) {
            for (Idx h = 0; h < h_in; h++) {
                for (Idx w = 0; w < w_in; w++) {
                    // CausalMask
                    for (Idx d = 0; d < d_in; d++) {
                        float in_value = in_0(b, h, w, d);

                        if (d > w + old_dim)
                            out_0(b, h, w, d) = in_value - MASK_INFINITY;
                        else
                            out_0(b, h, w, d) = in_value;
                    }
                }
            }
        }
    } else {
        auto in_ptr = in_0.raw_data_const();
        auto out_ptr = out_0.raw_data();
        memcpy(out_ptr, in_ptr, b_in * h_in * w_in * d_in * 4);
    }

    return GraphStatus::Success;
}

__attribute__((unused)) static float causalmaskCostFunc(const Op *op) {
    /*
     * add code here
     * */

    float cost = 0.0; // add cost computation here
    return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_CausalMask);