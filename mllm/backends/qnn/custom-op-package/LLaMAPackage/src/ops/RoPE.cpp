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

BEGIN_PKG_OP_DEFINITION(PKG_RoPE);

// op execute function declarations
template<typename TensorType, typename TensorType1>
GraphStatus ropeImpl(TensorType& out_0, const TensorType& in_0, const TensorType& sin, const TensorType& cos,
                     const TensorType1& h_cnt, const Tensor& pose_type);

// forward declaration of sample cost function
static float ropeCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((ropeImpl<Tensor>), "RoPE")
 */
DEF_PACKAGE_OP((ropeImpl<Tensor, Tensor>), "RoPE")

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
DEF_PACKAGE_PARAM_ORDER("RoPE", "pose_type", true, nullptr)

/* execute functions for ops */

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"
#include <stddef.h>

#define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
#define L2FETCH_AHEAD (BLOCK_SIZE)
#define ONE 0x3F800000
#define M_ONE 0xAF800000

int32_t hvx_rope_af(float* restrict input, float* restrict sin, float* restrict cos, float* restrict output, uint32_t size,
                    uint32_t partial_dimension) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr_half = (HVX_Vector*)(input + partial_dimension / 2);
  HVX_Vector* iptr2 = (HVX_Vector*)sin;
  HVX_Vector* iptr3 = (HVX_Vector*)cos;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_UVector* optr_half = (HVX_UVector*)(output + partial_dimension / 2);
  ;
  HVX_Vector sline1;
  HVX_Vector sline1_half;
  HVX_Vector sinline1p, sinline1c, sinline1;
  HVX_Vector cosline1p, cosline1c, cosline1;

  int32_t l2fetch_block;
  int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  int32_t leftover_size = leftover * sizeof(float);

  sinline1p = *iptr2++;
  cosline1p = *iptr3++;

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr3 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t d = 0; d < partial_dimension / 2; d += 32) {
      cosline1c = *iptr3++;
      cosline1 = Q6_V_valign_VVR(cosline1c, cosline1p, (size_t)cos);
      cosline1p = cosline1c;

      sinline1c = *iptr2++;
      sinline1 = Q6_V_valign_VVR(sinline1c, sinline1p, (size_t)sin);
      sinline1p = sinline1c;

      HVX_Vector* jiptr = iptr + d / 32;
      HVX_Vector* jiptr_half = iptr_half + d / 32;
      HVX_Vector* joptr = optr + d / 32;
      HVX_Vector* joptr_half = optr_half + d / 32;

      for (int32_t j = 0; j < size / partial_dimension; j++) {
        sline1 = *jiptr;
        sline1_half = *jiptr_half;

        // auto value = in_value * cos_value - in_value_2 * sin_value;
        {
          HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1, cosline1);
          HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1_half, sinline1);
          *joptr = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32));
        }

        // auto value2 = in_value * sin_value + in_value_2 * cos_value;
        {
          HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1_half, cosline1);
          HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1, sinline1);
          *joptr_half = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32));
        }

        jiptr += partial_dimension / 32;
        jiptr_half += partial_dimension / 32;
        joptr += partial_dimension / 32;
        joptr_half += partial_dimension / 32;
      }
    }
  }

  // if (vectors_in_rounddown > 0) {

  //   sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
  //   sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t) input);
  //   sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum,  Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

  // }

  if (leftover_size > 0) return -1;

  return 0;
}

static inline int32_t float_to_fp16s(float input) {
  union {
    int32_t i;
    __fp16 f[2];
  } fp32 = {.f = {(__fp16)input, (__fp16)input}};
  return fp32.i;
}

int32_t hvx_rope_uint8_af(uint8_t* restrict input, float* restrict sin, float* restrict cos, float* restrict output,
                          uint32_t size, uint32_t partial_dimension) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)sin;
  HVX_Vector* iptr3 = (HVX_Vector*)cos;
  HVX_UVector* optr = (HVX_UVector*)output;

  int32_t l2fetch_block;
  int32_t leftover = size & 127;
  int32_t vectors_in_rounddown = size / 128;
  int32_t leftover_size = leftover * sizeof(float);

  HVX_Vector zero_v_sf = Q6_V_vzero();
  uint32_t convert = 0x00800080;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);
  HVX_Vector one_vec = Q6_V_vsplat_R(float_to_fp16s(1.0));

  //
  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    //
    HVX_Vector sinline1_low = *iptr2;
    HVX_Vector cosline1_low = *iptr3;
    sinline1_low = Q6_Vqf32_vadd_VsfVsf(sinline1_low, Q6_V_vzero());
    cosline1_low = Q6_Vqf32_vadd_VsfVsf(cosline1_low, Q6_V_vzero());

    HVX_Vector sinline1_high = *(iptr2 + 1);
    HVX_Vector cosline1_high = *(iptr3 + 1);
    sinline1_high = Q6_Vqf32_vadd_VsfVsf(sinline1_high, Q6_V_vzero());
    cosline1_high = Q6_Vqf32_vadd_VsfVsf(cosline1_high, Q6_V_vzero());

    for (int32_t j = 0; j < size / partial_dimension; j++) {
      HVX_Vector sline1 = *iptr++;

      HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

      temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
      HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

      HVX_VectorPair result1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), one_vec);
      result1 = Q6_W_vshuff_VVR(Q6_V_hi_W(result1), Q6_V_lo_W(result1), -4);

      HVX_VectorPair result2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), one_vec);
      result2 = Q6_W_vshuff_VVR(Q6_V_hi_W(result2), Q6_V_lo_W(result2), -4);

      // auto value = in_value * cos_value - in_value_2 * sin_value;
      {
        HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result1), cosline1_low);
        HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result2), sinline1_low);
        *optr = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32));
      }

      // auto value2 = in_value * sin_value + in_value_2 * cos_value;
      {
        HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result2), cosline1_low);
        HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result1), sinline1_low);
        *(optr + 2) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32));
      }

      // auto value = in_value * cos_value - in_value_2 * sin_value;
      {
        HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result1), cosline1_high);
        HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result2), sinline1_high);
        *(optr + 1) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32));
      }

      // auto value2 = in_value * sin_value + in_value_2 * cos_value;
      {
        HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result2), cosline1_high);
        HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result1), sinline1_high);
        *(optr + 3) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32));
      }

      optr += 4;
    }
  }

  // if (vectors_in_rounddown > 0) {

  //   sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
  //   sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t) input);
  //   sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum,  Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

  // }

  if (leftover_size > 0) return -1;

  return 0;
}

int32_t hvx_rope_uint8_ahf(uint8_t* restrict input, float* restrict sin, float* restrict cos, __fp16* restrict output,
                           uint32_t size, uint32_t partial_dimension, float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)sin;
  HVX_Vector* iptr3 = (HVX_Vector*)cos;
  HVX_UVector* optr = (HVX_UVector*)output;

  int32_t l2fetch_block;
  int32_t leftover = size & 127;
  int32_t vectors_in_rounddown = size / 128;
  int32_t leftover_size = leftover * sizeof(float);

  HVX_Vector zero_v_sf = Q6_V_vzero();
  uint32_t convert = 0x00800080;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);

  HVX_Vector scale_vec = Q6_V_vsplat_R(float_to_fp16s(scale));

  //
  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    //
    HVX_Vector sinline1_low = *iptr2;
    HVX_Vector cosline1_low = *iptr3;
    sinline1_low = Q6_Vqf32_vadd_VsfVsf(sinline1_low, Q6_V_vzero());
    cosline1_low = Q6_Vqf32_vadd_VsfVsf(cosline1_low, Q6_V_vzero());

    HVX_Vector sinline1_high = *(iptr2 + 1);
    HVX_Vector cosline1_high = *(iptr3 + 1);
    sinline1_high = Q6_Vqf32_vadd_VsfVsf(sinline1_high, Q6_V_vzero());
    cosline1_high = Q6_Vqf32_vadd_VsfVsf(cosline1_high, Q6_V_vzero());

    for (int32_t j = 0; j < size / partial_dimension; j++) {
      HVX_Vector sline1 = *iptr++;

      HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

      temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
      HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

      HVX_VectorPair result1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), scale_vec);
      result1 = Q6_W_vshuff_VVR(Q6_V_hi_W(result1), Q6_V_lo_W(result1), -4);

      HVX_VectorPair result2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), scale_vec);
      result2 = Q6_W_vshuff_VVR(Q6_V_hi_W(result2), Q6_V_lo_W(result2), -4);

      {
        HVX_Vector first;
        HVX_Vector second;
        // auto value = in_value * cos_value - in_value_2 * sin_value;
        {
          HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result1), cosline1_low);
          HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result2), sinline1_low);
          first = Q6_Vqf32_vsub_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32);
        }

        // auto value = in_value * cos_value - in_value_2 * sin_value;
        {
          HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result1), cosline1_high);
          HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result2), sinline1_high);
          second = Q6_Vqf32_vsub_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32);
        }

        HVX_Vector r = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(second, first));
        r = Q6_Vh_vdeal_Vh(r);
        *optr = r;
      }

      {
        HVX_Vector first;
        HVX_Vector second;
        // auto value2 = in_value * sin_value + in_value_2 * cos_value;
        {
          HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result2), cosline1_low);
          HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result1), sinline1_low);
          first = Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32);
        }

        // auto value2 = in_value * sin_value + in_value_2 * cos_value;
        {
          HVX_Vector cos_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result2), cosline1_high);
          HVX_Vector sin_middle_value_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result1), sinline1_high);
          second = Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32, sin_middle_value_qf32);
        }
        HVX_Vector r = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(second, first));
        r = Q6_Vh_vdeal_Vh(r);
        *(optr + 1) = r;
      }

      optr += 2;
    }
  }

  // if (vectors_in_rounddown > 0) {

  //   sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
  //   sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t) input);
  //   sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum,  Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

  // }

  if (leftover_size > 0) return -1;

  return 0;
}

int32_t hvx_rope_ahf(__fp16* restrict input, float* restrict sin, float* restrict cos, __fp16* restrict output, uint32_t size,
                     uint32_t partial_dimension) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr_half = (HVX_Vector*)(input + partial_dimension / 2);
  HVX_Vector* iptr2 = (HVX_Vector*)sin;
  HVX_Vector* iptr3 = (HVX_Vector*)cos;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_UVector* optr_half = (HVX_UVector*)(output + partial_dimension / 2);
  ;
  HVX_Vector sline1;
  HVX_Vector sline1_half;

  int32_t l2fetch_block;
  int32_t leftover = size & 63;
  int32_t vectors_in_rounddown = size / 64;
  int32_t leftover_size = leftover * sizeof(float);

  HVX_Vector one_vsf = Q6_V_vsplat_R(ONE);
  HVX_Vector m_one_vqf32 = Q6_Vqf32_vsub_VsfVsf(Q6_V_vzero(), one_vsf);

  HVX_Vector one_vhf = Q6_V_vsplat_R(float_to_fp16s(1.0));
  // HVX_Vector m_one_vqf16 = Q6_Vqf32_vsub_VsfVsf(Q6_V_vzero(), one_vhf);

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr3 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t d = 0; d < partial_dimension / 2; d += 64) {
      HVX_Vector sinline1_low = *iptr2++;
      HVX_Vector cosline1_low = *iptr3++;

      HVX_Vector sinline1_high = *iptr2++;
      HVX_Vector cosline1_high = *iptr3++;

      HVX_Vector* jiptr = iptr + d / 64;
      HVX_Vector* jiptr_half = iptr_half + d / 64;
      HVX_Vector* joptr = optr + d / 64;
      HVX_Vector* joptr_half = optr_half + d / 64;

      for (int32_t j = 0; j < size / partial_dimension; j++) {
        sline1 = *jiptr;
        sline1_half = *jiptr_half;

        HVX_VectorPair sline1_half_pair = Q6_Wqf32_vmpy_VhfVhf(sline1_half, one_vhf);
        HVX_VectorPair sline1_pair = Q6_Wqf32_vmpy_VhfVhf(sline1, one_vhf);

        sline1_half_pair = Q6_W_vshuff_VVR(Q6_V_hi_W(sline1_half_pair), Q6_V_lo_W(sline1_half_pair), -4);
        sline1_pair = Q6_W_vshuff_VVR(Q6_V_hi_W(sline1_pair), Q6_V_lo_W(sline1_pair), -4);

        HVX_Vector m_sline1_half_low = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sline1_half_pair), m_one_vqf32);
        HVX_Vector m_sline1_half_hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sline1_half_pair), m_one_vqf32);

        // auto value = in_value * cos_value - in_value_2 * sin_value;
        HVX_Vector middle_value_low;
        {
          HVX_Vector cosline1_vqf32_low = Q6_Vqf32_vadd_VsfVsf(cosline1_low, Q6_V_vzero());
          HVX_Vector cos_middle_value_qf32_low = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sline1_pair), cosline1_vqf32_low);

          HVX_Vector sinline1_vqf32_low = Q6_Vqf32_vadd_VsfVsf(sinline1_low, Q6_V_vzero());

          HVX_Vector sin_middle_value_qf32_low = Q6_Vqf32_vmpy_Vqf32Vqf32(m_sline1_half_low, sinline1_vqf32_low);
          middle_value_low = Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32_low, sin_middle_value_qf32_low);
        }

        // auto value2 = in_value * sin_value + in_value_2 * cos_value;

        HVX_Vector middle_value_half_low;
        {
          HVX_Vector cosline1_vqf32_low = Q6_Vqf32_vadd_VsfVsf(cosline1_low, Q6_V_vzero());
          HVX_Vector cos_middle_value_qf32_low = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sline1_half_pair), cosline1_vqf32_low);

          HVX_Vector sinline1_vqf32_low = Q6_Vqf32_vadd_VsfVsf(sinline1_low, Q6_V_vzero());
          HVX_Vector sin_middle_value_qf32_low = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(sline1_pair), sinline1_vqf32_low);

          middle_value_half_low = Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32_low, sin_middle_value_qf32_low);
        }

        // second qf16 vector
        HVX_Vector middle_value_high;
        {
          HVX_Vector cosline1_vqf32_high = Q6_Vqf32_vadd_VsfVsf(cosline1_high, Q6_V_vzero());
          HVX_Vector cos_middle_value_qf32_high = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sline1_pair), cosline1_vqf32_high);

          HVX_Vector sinline1_vqf32_high = Q6_Vqf32_vadd_VsfVsf(sinline1_high, Q6_V_vzero());

          HVX_Vector sin_middle_value_qf32_high = Q6_Vqf32_vmpy_Vqf32Vqf32(m_sline1_half_hi, sinline1_vqf32_high);
          middle_value_high = Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32_high, sin_middle_value_qf32_high);
        }

        // auto value2 = in_value * sin_value + in_value_2 * cos_value;

        HVX_Vector middle_value_half_high;
        {
          HVX_Vector cosline1_vqf32_high = Q6_Vqf32_vadd_VsfVsf(cosline1_high, Q6_V_vzero());
          HVX_Vector cos_middle_value_qf32_high = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sline1_half_pair), cosline1_vqf32_high);

          HVX_Vector sinline1_vqf32_high = Q6_Vqf32_vadd_VsfVsf(sinline1_high, Q6_V_vzero());
          HVX_Vector sin_middle_value_qf32_high = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(sline1_pair), sinline1_vqf32_high);

          middle_value_half_high = Q6_Vqf32_vadd_Vqf32Vqf32(cos_middle_value_qf32_high, sin_middle_value_qf32_high);
        }

        HVX_Vector sline = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(middle_value_high, middle_value_low));
        sline = Q6_Vh_vdeal_Vh(sline);

        HVX_Vector sline_half = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(middle_value_half_high, middle_value_half_low));
        sline_half = Q6_Vh_vdeal_Vh(sline_half);

        *joptr = sline;
        *joptr_half = sline_half;

        jiptr += partial_dimension / 64;
        jiptr_half += partial_dimension / 64;
        joptr += partial_dimension / 64;
        joptr_half += partial_dimension / 64;
      }
    }
  }

  // if (vectors_in_rounddown > 0) {

  //   sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
  //   sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t) input);
  //   sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum,  Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

  // }

  if (leftover_size > 0) return -1;

  return 0;
}

template<typename TensorType, typename TensorType1>
GraphStatus ropeImpl(TensorType& out_0, const TensorType& in_0, const TensorType& sin, const TensorType& cos,
                     const TensorType1& h_cnt, const Tensor& pose_type) {
  out_0.set_dims(in_0);

  auto pose_type_ = pose_type(0, 0, 0, 0);
  auto h_cnt_ = static_cast<uint32_t>(h_cnt(0, 0, 0, 0));

  if (pose_type_ == 4) {
    DType dtype = out_0.get_dtype();

    if (in_0.get_dtype() == DType::Float32 && dtype == DType::Float32) {
      auto in_ptr = (float*)in_0.raw_data_const();
      auto sin_ptr = (float*)sin.raw_data_const();
      auto cos_ptr = (float*)cos.raw_data_const();
      auto out_ptr = (float*)out_0.raw_data();

      auto [b_in, h_in, w_in, d_in] = in_0.dims();

      uint32_t half_dimension = d_in / 2;
      sin_ptr += half_dimension * h_cnt_;
      cos_ptr += half_dimension * h_cnt_;

      int partial_dimension = d_in;

      // NSHD
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          // for (Idx w = 0; w < w_in; w++) {
          hvx_rope_af(in_ptr, sin_ptr, cos_ptr, out_ptr, w_in * d_in, partial_dimension);

          in_ptr += w_in * d_in;
          out_ptr += w_in * d_in;
          // }

          sin_ptr += half_dimension;
          cos_ptr += half_dimension;
        }
      }
    } else if (in_0.get_dtype() == DType::Float16 && dtype == DType::Float16) {
      auto in_ptr = (__fp16*)in_0.raw_data_const();
      auto sin_ptr = (float*)sin.raw_data_const();
      auto cos_ptr = (float*)cos.raw_data_const();
      auto out_ptr = (__fp16*)out_0.raw_data();

      auto [b_in, h_in, w_in, d_in] = in_0.dims();

      uint32_t half_dimension = d_in / 2;
      sin_ptr += half_dimension * h_cnt_;
      cos_ptr += half_dimension * h_cnt_;

      int partial_dimension = d_in;

      // NSHD
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          // for (Idx w = 0; w < w_in; w++) {
          hvx_rope_ahf(in_ptr, sin_ptr, cos_ptr, out_ptr, w_in * d_in, partial_dimension);

          in_ptr += w_in * d_in;
          out_ptr += w_in * d_in;
          // }

          sin_ptr += half_dimension;
          cos_ptr += half_dimension;
        }
      }
    } else if (in_0.get_dtype() == DType::QUInt8 && dtype == DType::Float32) {
      auto in_ptr = (uint8_t*)in_0.raw_data_const();
      auto sin_ptr = (float*)sin.raw_data_const();
      auto cos_ptr = (float*)cos.raw_data_const();
      auto out_ptr = (float*)out_0.raw_data();

      auto [b_in, h_in, w_in, d_in] = in_0.dims();

      uint32_t half_dimension = d_in / 2;
      sin_ptr += half_dimension * h_cnt_;
      cos_ptr += half_dimension * h_cnt_;

      int partial_dimension = d_in;

      // NSHD
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          // for (Idx w = 0; w < w_in; w++) {
          hvx_rope_uint8_af(in_ptr, sin_ptr, cos_ptr, out_ptr, w_in * d_in, partial_dimension);

          in_ptr += w_in * d_in;
          out_ptr += w_in * d_in;
          // }

          sin_ptr += half_dimension;
          cos_ptr += half_dimension;
        }
      }
    } else if (in_0.get_dtype() == DType::QUInt8 && dtype == DType::Float16) {
      auto in_ptr = (uint8_t*)in_0.raw_data_const();
      auto sin_ptr = (float*)sin.raw_data_const();
      auto cos_ptr = (float*)cos.raw_data_const();
      auto out_ptr = (__fp16*)out_0.raw_data();

      float scale_ = in_0.interface_scale();

      auto [b_in, h_in, w_in, d_in] = in_0.dims();

      uint32_t half_dimension = d_in / 2;
      sin_ptr += half_dimension * h_cnt_;
      cos_ptr += half_dimension * h_cnt_;

      int partial_dimension = d_in;

      // NSHD
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          // for (Idx w = 0; w < w_in; w++) {
          hvx_rope_uint8_ahf(in_ptr, sin_ptr, cos_ptr, out_ptr, w_in * d_in, partial_dimension, scale_);

          in_ptr += w_in * d_in;
          out_ptr += w_in * d_in;
          // }

          sin_ptr += half_dimension;
          cos_ptr += half_dimension;
        }
      }
    }

  } else {
    // only support pose_type == 2 (LLaMA) now
    return GraphStatus::ErrorFatal;
  }

  return GraphStatus::Success;
}

#else

template<typename TensorType, typename TensorType1>
GraphStatus ropeImpl(TensorType& out_0, const TensorType& in_0, const TensorType& sin, const TensorType& cos,
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

  debuglog("RoPE execute... dims=(%zdx%zdx%zdx%zd)", in_0.dim(0), in_0.dim(1), in_0.dim(2), in_0.dim(3));
  debuglog("RoPE execute... dims=(%zdx%zdx%zdx%zd)", sin.dim(0), sin.dim(1), sin.dim(2), sin.dim(3));
  debuglog("RoPE execute... dims=(%zdx%zdx%zdx%zd)", cos.dim(0), cos.dim(1), cos.dim(2), cos.dim(3));

  // BSHD =>  NHWC

  // Todo: We need consider to store the sequence position if we have KV Cache

  auto pose_type_ = pose_type(0, 0, 0, 0);
  auto h_cnt_ = static_cast<uint32_t>(h_cnt(0, 0, 0, 0));

  out_0.set_dims(in_0);
  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  if (pose_type_ == 4) {
    DType dtype = out_0.get_dtype();

    if (dtype == DType::Float32) {
      for (Idx b = 0; b < b_in; b++) {
        for (Idx h = 0; h < h_in; h++) {
          for (Idx w = 0; w < w_in; w++) {
            int s = h;  //  BSHD order
            int partial_dimension = d_in;
            int half = (int)(partial_dimension / 2);
            for (Idx d = 0; d < partial_dimension / 2; ++d) {
              float in_value = in_0(b, h, w, d);
              float in_value_2 = in_0(b, h, w, d + half);
              float sin_value = sin(0, 0, s + h_cnt_, d);
              float cos_value = cos(0, 0, s + h_cnt_, d);
              auto value = in_value * cos_value - in_value_2 * sin_value;
              auto value2 = in_value * sin_value + in_value_2 * cos_value;
              out_0(b, h, w, d) = value;
              out_0(b, h, w, d + half) = value2;
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
        for (Idx h = 0; h < h_in; h++) {
          for (Idx w = 0; w < w_in; w++) {
            int s = h;  //  BSHD order
            int partial_dimension = d_in;
            int half = (int)(partial_dimension / 2);
            for (Idx d = 0; d < partial_dimension / 2; ++d) {
              __fp16 in_value = *in_ptr;
              __fp16 in_value_2 = *(in_ptr + half);
              float sin_value = sin(0, 0, s + h_cnt_, d);
              float cos_value = cos(0, 0, s + h_cnt_, d);
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
  }

  return GraphStatus::Success;
}

#endif

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
END_PKG_OP_DEFINITION(PKG_RoPE);