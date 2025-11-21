//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_QLayerNorm);

// op execute function declarations
template<typename TensorType>
GraphStatus qlayernormImpl(TensorType& out_0, const TensorType& in_0, const TensorType& weights, const TensorType& bias);

// forward declaration of sample cost function
static float qlayernormCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((qlayernormImpl<Tensor>), "QLayerNorm")
 */
DEF_PACKAGE_OP((qlayernormImpl<Tensor>), "QLayerNorm")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((qlayernormImpl<PlainFloatTensor>), "QLayerNorm", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((qlayernormImpl<PlainFloatTensor>),
 * "QLayerNorm", qlayernormCostFunc, Flags::RESOURCE_HVX)
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

#include <hexagon_types.h>
#include "hvx_internal.h"
#include <stddef.h>

#define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
#define L2FETCH_AHEAD (BLOCK_SIZE)

int32_t hvx_qlayernorm_af(float* restrict input, float* restrict weights, float* restrict bias, float* restrict output,
                          uint32_t size) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)weights;
  HVX_Vector* iptr3 = (HVX_Vector*)bias;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;

  HVX_Vector zero;

  float __attribute__((aligned(VLEN))) tmp_buf[32];
  int32_t block, l2fetch_block;
  int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  int32_t leftover_size = leftover * sizeof(float);

  zero = Q6_V_vzero();

  // sline1p = *iptr++;

  // x sum
  HVX_Vector xsum = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
      xsum = Q6_Vqf32_vadd_Vqf32Vqf32(xsum, sline1);

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
    xsum = Q6_Vqf32_vadd_Vqf32Vqf32(xsum, sline1);
  }

  union {
    float f;
    uint32_t ui;
  } mean_value;
  mean_value.f = 0.0f;

  for (int32_t i = 64; i >= 4; i >>= 1) { xsum = Q6_Vqf32_vadd_Vqf32Vqf32(xsum, Q6_V_vlalign_VVR(xsum, zero, i)); }

  xsum = Q6_Vsf_equals_Vqf32(xsum);
  *(HVX_Vector*)tmp_buf = xsum;

  mean_value.f = xsum[31] / size;

  // x-e^2 sum
  iptr = (HVX_Vector*)input;
  sline1p = *iptr++;

  HVX_Vector x2sum = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());

  HVX_Vector mean_vsf = Q6_V_vsplat_R(mean_value.ui);

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sline1 = Q6_Vqf32_vsub_Vqf32Vqf32(sline1, mean_vsf);
      x2sum = Q6_Vqf32_vadd_Vqf32Vqf32(x2sum, Q6_Vqf32_vmpy_Vqf32Vqf32(sline1, sline1));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    sline1 = Q6_Vqf32_vsub_Vqf32Vqf32(sline1, mean_vsf);
    x2sum = Q6_Vqf32_vadd_Vqf32Vqf32(x2sum, Q6_Vqf32_vmpy_Vqf32Vqf32(sline1, sline1));
  }

  float epsilon_ = 1e-5;
  union {
    float f;
    uint32_t ui;
  } sum_value;
  sum_value.f = 0.0f;

  for (int32_t i = 64; i >= 4; i >>= 1) { x2sum = Q6_Vqf32_vadd_Vqf32Vqf32(x2sum, Q6_V_vlalign_VVR(x2sum, zero, i)); }

  x2sum = Q6_Vsf_equals_Vqf32(x2sum);
  *(HVX_Vector*)tmp_buf = x2sum;

  sum_value.f = 1.0f / sqrtf(x2sum[31] / size + epsilon_);

  // x * 1/rsqrt(sum)
  iptr = (HVX_Vector*)input;
  sline1p = *iptr++;
  sline2p = *iptr2++;
  sline3p = *iptr3++;

  HVX_Vector irsqrt_vsf = Q6_V_vsplat_R(sum_value.ui);
  HVX_Vector irsqrt_vqf32 = Q6_Vqf32_vadd_VsfVsf(irsqrt_vsf, Q6_V_vzero());

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr3 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline2c = *iptr2++;
      sline3c = *iptr3++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)weights);
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)bias);

      sline1 = Q6_Vqf32_vsub_Vqf32Vqf32(sline1, mean_vsf);

      HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(sline1, sline2);
      middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);
      middle_value_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(middle_value_qf32, sline3);

      *optr++ = Q6_Vsf_equals_Vqf32(middle_value_qf32);

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    sline2c = is_aligned(iptr2, VLEN) && leftover == 0 ? sline2p : *iptr2++;
    sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)weights);

    sline3c = is_aligned(iptr3, VLEN) && leftover == 0 ? sline3p : *iptr3++;
    sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)weights);

    sline1 = Q6_Vqf32_vsub_VsfVsf(sline1, mean_vsf);

    HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(sline1, sline2);
    middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);
    middle_value_qf32 = Q6_Vqf32_vadd_Vqf32Vqf32(middle_value_qf32, sline3);

    *optr++ = Q6_Vsf_equals_Vqf32(middle_value_qf32);
  }

  if (leftover_size > 0) return -1;

  return 0;
}

template<typename TensorType>
GraphStatus qlayernormImpl(TensorType& out_0, const TensorType& in_0, const TensorType& weights, const TensorType& bias)

{
  out_0.set_dims(in_0);

  // NHWC

  auto in_ptr = (float*)in_0.raw_data_const();
  auto out_ptr = (float*)out_0.raw_data();
  auto weights_ptr = (float*)weights.raw_data_const();
  auto bias_ptr = (float*)bias.raw_data_const();

  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // RMS
        hvx_qlayernorm_af(in_ptr, weights_ptr, bias_ptr, out_ptr, d_in);

        in_ptr += d_in;
        out_ptr += d_in;
      }
    }
  }

  return GraphStatus::Success;
}

#else

template<typename TensorType>
GraphStatus qlayernormImpl(TensorType& out_0, const TensorType& in_0, const TensorType& weights, const TensorType& bias)

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
  return GraphStatus::Success;
}

#endif

__attribute__((unused)) static float qlayernormCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_QLayerNorm);