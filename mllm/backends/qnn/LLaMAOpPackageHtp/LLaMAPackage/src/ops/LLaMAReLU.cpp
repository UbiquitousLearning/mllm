//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_LLaMAReLU);

// op execute function declarations
template <typename TensorType>
GraphStatus llamareluImpl(TensorType &out_0,
                          const TensorType &in_0);

// forward declaration of sample cost function
static float llamareluCostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((llamareluImpl<Tensor>), "LLaMAReLU")
 */
DEF_PACKAGE_OP((llamareluImpl<Tensor>), "LLaMAReLU")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((llamareluImpl<PlainFloatTensor>), "LLaMAReLU", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((llamareluImpl<PlainFloatTensor>),
 * "LLaMAReLU", llamareluCostFunc, Flags::RESOURCE_HVX)
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

// #ifndef REFERENCE_OP

// #include "qhmath_hvx.h"
// #include "hvx_internal.h"
// #include <hexagon_types.h>
// #include <stddef.h>

// #define BLOCK_SIZE       (8*1024/VLEN)  /* vector chunks */
// #define L2FETCH_AHEAD    (BLOCK_SIZE)
// #define ONE      0x3F800000
// #define M_ONE    0xAF800000

// int32_t hvx_relu_au8(uint8_t *restrict input, uint8_t *restrict output, uint32_t size)
// {
//     HVX_Vector *input_v_ptr;
//     HVX_UVector *output_v_ptr;
//     HVX_Vector slinep;
//     HVX_Vector slinec;
//     HVX_Vector sline;
//     int32_t block, l2fetch_block;
//     int32_t leftover = size & 128;
//     int32_t vectors_in_rounddown = size / 128;
//     int32_t leftover_size = leftover * sizeof(uint8_t);

//     /* Check input arguments. Return error status if some argument has invalid value */
//     if ((input == 0) || (output == 0) || (size == 0))
//     {
//         return -1;
//     }

//     input_v_ptr = (HVX_Vector *) input;
//     output_v_ptr = (HVX_UVector *) output;

//     HVX_Vector vO  = Q6_Vb_vsplat_R(0x80808080u);

//     /*
//      * If input data is not aligned to HVX vector size, compose aligned vectors
//      * from data loaded in slinep and slinec
//      */
//     slinep = *input_v_ptr++;

//     /*
//      * Handle number of whole vectors in input data.
//      * Don't process last vector in order to avoid out-of-boundary load.
//      */
//     for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE)
//     {
//         block = Q6_R_min_RR(i, BLOCK_SIZE);
//         l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

//         if (l2fetch_block > 0)
//         {
//             l2fetch(input_v_ptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
//         }

//         /* Process one vector at a time */
//         for (int32_t j = 0; j < block; ++j)
//         {
//             slinec = *input_v_ptr++;

//             /* Compose vector of input data from slinec and slinep */
//             sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

//             /* Store results to the output buffer and convert from qf32 to sf */
//             *((HVX_UVector *)(output_v_ptr++)) =  Q6_Vub_vmax_VubVub(vO, sline);

//             /* Prepare slinep for next iteration */
//             slinep = slinec;
//         }
//     }

//     /* Handle last whole vector from input data */
//     if (vectors_in_rounddown > 0)
//     {
//         slinec = is_aligned(input_v_ptr, VLEN) && leftover == 0 ? slinep : *input_v_ptr++;
//         sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

//         /* Convert from qf32 to sf, store output and go to handle leftover */
//         *((HVX_UVector *)(output_v_ptr++)) = Q6_Vub_vmax_VubVub(vO, sline);

//         slinep = slinec;
//     }

//     /* Handle leftover elements */
//     if (leftover > 0)
//     {
//         slinec = (is_in_one_chunk(input_v_ptr, leftover_size, VLEN)
//                    ? slinep
//                    : *input_v_ptr++);

//         sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

//         /* Store output */
//         vstu_variable(output_v_ptr, leftover_size, Q6_Vub_vmax_VubVub(vO, sline));
//     }

//     return 0;
// }

// template<typename TensorType>
// GraphStatus llamareluImpl(TensorType& out_0,
//                           const TensorType& in_0)

// {
//   /*
//    * add code here
//    * */
//   /*
//    * To have good performance and stability, it is required to avoid heap memory
//    * allocation in this function. The heap memory allocation includes but not
//    * limited to calling malloc, operator new, constructing STL container objects
//    * like std::vector with default allocator, and adding items like calling
//    * std::vector::push_back to STL container objects with default allocator.
//    *
//    * Please check in SDK documentation for more information.
//    */

//   out_0.set_dims(in_0);

//   const auto [bIn, hIn, wIn, dIn] = in_0.dims();

//   auto in_ptr = (uint8_t*)in_0.raw_data_const();
//   auto out_ptr = (uint8_t*)out_0.raw_data();

//   hvx_relu_au8(out_ptr, in_ptr, bIn * hIn * wIn * dIn * sizeof (uint8_t));

//   return GraphStatus::Success;
// }
// #else
template <typename TensorType>
GraphStatus llamareluImpl(TensorType &out_0,
                          const TensorType &in_0)

{
    out_0.set_dims(in_0);
    // NHWC

    if (in_0.get_dtype() == DType::QUInt8) {
        auto [b_in, h_in, w_in, d_in] = in_0.dims();
        for (Idx b = 0; b < b_in; b++) {
            for (Idx h = 0; h < h_in; h++) {
                for (Idx w = 0; w < w_in; w++) {
                    // SiLU
                    for (Idx d = 0; d < d_in; d++) {
                        uint8_t inval = in_0(b, h, w, d);
                        if (inval < 0)
                            inval = 0;

                        out_0(b, h, w, d) = inval;
                    }
                }
            }
        }
    } else if (in_0.get_dtype() == DType::Float16) {
        auto [b_in, h_in, w_in, d_in] = in_0.dims();

        auto out_ptr = (__fp16 *)out_0.raw_data();
        auto in_ptr = (__fp16 *)in_0.raw_data_const();

        for (Idx b = 0; b < b_in; b++) {
            for (Idx h = 0; h < h_in; h++) {
                for (Idx w = 0; w < w_in; w++) {
                    for (Idx d = 0; d < d_in; d++) {
                        __fp16 inval = *in_ptr++;
                        if (inval < 0)
                            inval = 0;

                        *out_ptr++ = inval;
                    }
                }
            }
        }
    } else if (in_0.get_dtype() == DType::Float32) {
        auto [b_in, h_in, w_in, d_in] = in_0.dims();
        for (Idx b = 0; b < b_in; b++) {
            for (Idx h = 0; h < h_in; h++) {
                for (Idx w = 0; w < w_in; w++) {
                    for (Idx d = 0; d < d_in; d++) {
                        float inval = in_0(b, h, w, d);
                        if (inval < 0)
                            inval = 0;

                        out_0(b, h, w, d) = inval;
                    }
                }
            }
        }
    }

    return GraphStatus::Success;
}

// #endif

__attribute__((unused)) static float llamareluCostFunc(const Op *op) {
    /*
     * add code here
     * */

    float cost = 0.0; // add cost computation here
    return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_LLaMAReLU);