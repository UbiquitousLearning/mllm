//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_RoPESimple);

// op execute function declarations
template<typename TensorType>
GraphStatus ropeSimpleImpl(TensorType& out_0, const TensorType& in_0, const TensorType& sin, const TensorType& cos);

// forward declaration of sample cost function
static float ropeSimpleCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((ropeImpl<Tensor>), "RoPE")
 */
DEF_PACKAGE_OP((ropeSimpleImpl<Tensor>), "RoPESimple")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((ropeImpl<PlainFloatTensor>), "RoPE", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((ropeImpl<PlainFloatTensor>),
 * "RoPE", ropeCostFunc, Flags::RESOURCE_HVX)
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

// #define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
// #define L2FETCH_AHEAD (BLOCK_SIZE)
// #define ONE 0x3F800000
// #define M_ONE 0xAF800000

// // TODO: hvx ropesimple implementation

// template <typename TensorType>
// GraphStatus ropeSimpleImpl(TensorType &out_0,
//                      const TensorType &in_0,
//                      const TensorType &sin,
//                      const TensorType &cos) {
//     out_0.set_dims(in_0);

//     return GraphStatus::Success;
// }

// #else

template<typename TensorType>
GraphStatus ropeSimpleImpl(TensorType& out_0, const TensorType& in_0, const TensorType& sin, const TensorType& cos)

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

  debuglog("RoPESimple execute... dims=(%zdx%zdx%zdx%zd)", in_0.dim(0), in_0.dim(1), in_0.dim(2), in_0.dim(3));
  debuglog("RoPESimple execute... dims=(%zdx%zdx%zdx%zd)", sin.dim(0), sin.dim(1), sin.dim(2), sin.dim(3));
  debuglog("RoPESimple execute... dims=(%zdx%zdx%zdx%zd)", cos.dim(0), cos.dim(1), cos.dim(2), cos.dim(3));

  // BSHD =>  NHWC

  out_0.set_dims(in_0);
  auto [b_in, w_in, h_in, d_in] = in_0.dims();
  DType dtype = out_0.get_dtype();

  if (dtype == DType::Float32) {
    for (Idx b = 0; b < b_in; b++) {
      for (Idx w = 0; w < w_in; w++) {
        for (Idx h = 0; h < h_in; h++) {
          int partial_dimension = d_in;
          int half = (int)(partial_dimension / 2);
          for (Idx d = 0; d < partial_dimension / 2; ++d) {
            float in_value = in_0(b, w, h, d);
            float in_value_2 = in_0(b, w, h, d + half);
            float sin_value = sin(0, 0, w, d);
            float cos_value = cos(0, 0, w, d);
            auto value = in_value * cos_value - in_value_2 * sin_value;
            auto value2 = in_value * sin_value + in_value_2 * cos_value;
            out_0(b, w, h, d) = value;
            out_0(b, w, h, d + half) = value2;
          }
        }
      }
    }
  } else if (dtype == DType::Float16) {
    auto in_ptr = (__fp16*)in_0.raw_data_const();
    // auto sin_ptr = (__fp16*)sin.raw_data_const();
    // auto cos_ptr = (__fp16*)cos.raw_data_const();
    auto out_ptr = (__fp16*)out_0.raw_data();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx w = 0; w < w_in; w++) {
        for (Idx h = 0; h < h_in; h++) {
          int partial_dimension = d_in;
          int half = (int)(partial_dimension / 2);
          for (Idx d = 0; d < partial_dimension / 2; ++d) {
            __fp16 in_value = *in_ptr;
            __fp16 in_value_2 = *(in_ptr + half);
            float sin_value = sin(0, 0, w, d);
            float cos_value = cos(0, 0, w, d);
            auto value = in_value * cos_value - in_value_2 * sin_value;
            auto value2 = in_value * sin_value + in_value_2 * cos_value;
            *out_ptr = static_cast<__fp16>(value);
            *(out_ptr + half) = static_cast<__fp16>(value2);

            out_ptr++;
            in_ptr++;
          }

          out_ptr += half;
          in_ptr += half;
        }
      }
    }
  }
  return GraphStatus::Success;
}

// #endif

__attribute__((unused)) static float ropeCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_RoPESimple);