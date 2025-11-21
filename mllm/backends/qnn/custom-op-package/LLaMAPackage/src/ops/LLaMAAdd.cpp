//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_LLaMAAdd);

// op execute function declarations
template<typename TensorType>
GraphStatus llamaaddImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1);

// forward declaration of sample cost function
static float llamaaddCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((llamaaddImpl<Tensor>), "LLaMAAdd")
 */
DEF_PACKAGE_OP((llamaaddImpl<Tensor>), "LLaMAAdd")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((llamaaddImpl<PlainFloatTensor>), "LLaMAAdd", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((llamaaddImpl<PlainFloatTensor>),
 * "LLaMAAdd", llamaaddCostFunc, Flags::RESOURCE_HVX)
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

#ifndef REFERENCE_OP

#include "hvx_internal.h"
#include <hexagon_types.h>
#include <stddef.h>

#define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
#define L2FETCH_AHEAD (BLOCK_SIZE)

int32_t hvx_add_af(float* restrict input, float* restrict input2, float* restrict output, uint32_t size) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)input2;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;

  // HVX_Vector v128 = Q6_Vb_vsplat_R(0x80808080u);

  int32_t block, l2fetch_block;
  int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr2++;

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline2c = *iptr2++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input2);

      // Our add consider uint8->int8 bugs from QNN.
      // sline2 = Q6_Vb_vsub_VbVb(sline2, v128);
      *optr++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sline1, sline2));

      sline1p = sline1c;
      sline2p = sline2c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    sline2c = is_aligned(iptr2, VLEN) && leftover == 0 ? sline2p : *iptr2++;
    sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input2);

    // sline2 = Q6_Vb_vsub_VbVb(sline2, v128);
    *optr++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sline1, sline2));
  }

  // Handle leftover elements.
  if (leftover_size > 0) {
    sline1c = (is_in_one_chunk(iptr, leftover_size, VLEN) ? sline1p : *iptr++);
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    sline2c = (is_in_one_chunk(iptr2, leftover_size, VLEN) ? sline2p : *iptr2++);
    sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input2);

    // sline2 = Q6_Vb_vsub_VbVb(sline2, v128);
    vstu_variable(optr, leftover_size, Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(sline1, sline2)));
  }

  return 0;
}

template<typename TensorType>
GraphStatus llamaaddImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1)

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

  auto in_ptr = (float*)in_0.raw_data_const();
  auto in2_ptr = (float*)in_1.raw_data_const();
  auto out_ptr = (float*)out_0.raw_data();

  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  size_t size = b_in * h_in * w_in * d_in;

  hvx_add_af(in_ptr, in2_ptr, out_ptr, size);

  return GraphStatus::Success;
}

#else

template<typename TensorType>
GraphStatus llamaaddImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1)

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

  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // mul
        for (Idx d = 0; d < d_in; d++) {
          float inval = in_0(b, h, w, d);
          float inval2 = in_1(b, h, w, d);
          float outval = inval + inval2;

          out_0(b, h, w, d) = outval;
        }
      }
    }
  }

  return GraphStatus::Success;
}

#endif

__attribute__((unused)) static float llamaaddCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_LLaMAAdd);