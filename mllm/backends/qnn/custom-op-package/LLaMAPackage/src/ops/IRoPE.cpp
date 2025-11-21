//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"
#include "HTP/core/tensor.h"

BEGIN_PKG_OP_DEFINITION(PKG_IRoPE);

// op execute function declarations
template<typename TensorType, typename TensorType1>
GraphStatus iropeImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1, const TensorType& cos,
                      const TensorType1& h_cnt, const Tensor& pose_type);

// forward declaration of sample cost function
static float iropeCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((iropeImpl<Tensor, Tensor>), "IRoPE")
 */
DEF_PACKAGE_OP((iropeImpl<Tensor, Tensor>), "IRoPE")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((iropeImpl<PlainFloatTensor, PlainFloatTensor>), "IRoPE", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((iropeImpl<PlainFloatTensor, PlainFloatTensor>),
 * "IRoPE", iropeCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("IRoPE", "pose_type", true, nullptr)

/* execute functions for ops */

template<typename TensorType, typename TensorType1>
GraphStatus iropeImpl(TensorType& out_0, const TensorType& in_0, const TensorType& sin, const TensorType& cos,
                      const TensorType1& h_cnt, const Tensor& pose_type)

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
  auto pose_type_ = pose_type(0, 0, 0, 0);
  auto h_cnt_ = static_cast<uint32_t>(h_cnt(0, 0, 0, 0));

  out_0.set_dims(in_0);
  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  uint32_t half_dimension = d_in / 2;

  auto sin_ptr = (uint8_t*)sin.raw_data_const();
  auto cos_ptr = (uint8_t*)cos.raw_data_const();

  auto in_ptr = (uint8_t*)in_0.raw_data_const();

  sin_ptr += half_dimension * h_cnt_;
  cos_ptr += half_dimension * h_cnt_;

  // float scale_ = in_0.interface_scale() * sin.interface_scale() * cos.interface_scale();

  if (pose_type_ == 4) {
    DType dtype = out_0.get_dtype();

    if (dtype == DType::Float32) {
      auto out_ptr = (float*)out_0.raw_data();
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          for (Idx w = 0; w < w_in; w++) {
            int partial_dimension = d_in;
            for (Idx d = 0; d < partial_dimension / 2; ++d) {
              int in_value = *in_ptr;
              int in_value_2 = *(in_ptr + half_dimension);

              int sin_value = *(sin_ptr + d);
              int cos_value = *(cos_ptr + d);
              float value = (in_value - 128) * (cos_value - 128) * cos.interface_scale()
                            - (in_value_2 - 128) * (sin_value - 128) * sin.interface_scale();
              float value2 = (in_value - 128) * (sin_value - 128) * sin.interface_scale()
                             + (in_value_2 - 128) * (cos_value - 128) * cos.interface_scale();

              *out_ptr = value;
              *(out_ptr + half_dimension) = value2;

              out_ptr++;
              in_ptr++;
            }

            in_ptr += half_dimension;
            out_ptr += half_dimension;
          }

          sin_ptr += half_dimension;
          cos_ptr += half_dimension;
        }
      }
    } else if (dtype == DType::Float16) {
      auto out_ptr = (__fp16*)out_0.raw_data();
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          for (Idx w = 0; w < w_in; w++) {
            int partial_dimension = d_in;
            for (Idx d = 0; d < partial_dimension / 2; ++d) {
              int in_value = *in_ptr;
              int in_value_2 = *(in_ptr + half_dimension);

              int sin_value = *(sin_ptr + d);
              int cos_value = *(cos_ptr + d);
              float value = (in_value - 128) * (cos_value - 128) * cos.interface_scale()
                            - (in_value_2 - 128) * (sin_value - 128) * sin.interface_scale();
              float value2 = (in_value - 128) * (sin_value - 128) * sin.interface_scale()
                             + (in_value_2 - 128) * (cos_value - 128) * cos.interface_scale();

              *out_ptr = static_cast<__fp16>(value);
              *(out_ptr + half_dimension) = static_cast<__fp16>(value2);

              out_ptr++;
              in_ptr++;
            }

            in_ptr += half_dimension;
            out_ptr += half_dimension;
          }

          sin_ptr += half_dimension;
          cos_ptr += half_dimension;
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float iropeCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_IRoPE);