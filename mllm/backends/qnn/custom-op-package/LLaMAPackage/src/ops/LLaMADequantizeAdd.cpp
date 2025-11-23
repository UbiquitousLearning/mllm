//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_LLaMADequantizeAdd);

// op execute function declarations
template<typename TensorType, typename TensorType1, typename TensorType2, typename TensorType3>
GraphStatus llamadequantizeaddImpl(TensorType1& out_0, const TensorType1& in_0, const TensorType& in_1,
                                   const PlainFloatTensor& scale);

// forward declaration of sample cost function
static float llamadequantizeaddCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((llamadequantizeaddImpl<Tensor, Tensor, Tensor, Tensor>), "LLaMADequantizeAdd")
 */
DEF_PACKAGE_OP((llamadequantizeaddImpl<Tensor, Tensor, Tensor, Tensor>), "LLaMADequantizeAdd")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((llamadequantizeaddImpl<PlainFloatTensor, PlainFloatTensor, PlainFloatTensor,
 * PlainFloatTensor>), "LLaMADequantizeAdd", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((llamadequantizeaddImpl<PlainFloatTensor, PlainFloatTensor, PlainFloatTensor,
 * PlainFloatTensor>), "LLaMADequantizeAdd", llamadequantizeaddCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("LLaMADequantizeAdd", "scale", true, nullptr)

/* execute functions for ops */
#ifndef REFERENCE_OP
#include <hexagon_types.h>
#include "hvx_internal.h"
#include <stddef.h>

#define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
#define L2FETCH_AHEAD (BLOCK_SIZE)

static inline int32_t float_to_fp16s(float input) {
  union {
    int32_t i;
    __fp16 f[2];
  } fp32 = {.f = {(__fp16)input, (__fp16)input}};
  return fp32.i;
}

static HVX_INLINE_ALWAYS uint32_t float_to_bits(float x) {
  union {
    float f;
    uint32_t i;
  } fp32 = {.f = x};
  return fp32.i;
}

/* execute functions for ops */
int32_t qhmath_hvx_dequantize_add_ahf(int8_t* restrict input, float_t* restrict bias, int8_t* restrict output, uint32_t size,
                                      float scale) {
  if ((input == NULL) || (bias == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* bptr = (HVX_Vector*)bias;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector scale_vec;

  int32_t block, l2fetch_block;
  int32_t leftover = size & 127;
  int32_t vectors_in_rounddown = size / 128;  // element number!
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  uint32_t convert = 0x00800080;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);

  scale_vec = Q6_V_vsplat_R(float_to_fp16s(scale));
  HVX_Vector zero_v_sf = Q6_V_vzero();

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
      HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

      temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
      HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

      HVX_Vector bvec1 = *bptr++;  // load 32 float elements
      HVX_Vector bvec2 = *bptr++;
      // see HVX documention for Vector shuffle and deal cross-lane
      // Q6_Vhf_equals_Wqf32 will use elements in the lower vector as the odd elements, so need to transpose here
      HVX_VectorPair bias_pair =
          Q6_W_vdeal_VVR(Q6_Vqf32_equals_Vsf(bvec2), Q6_Vqf32_equals_Vsf(bvec1), -4);  // make a fp32 pair
      HVX_Vector hf16_bias = Q6_Vhf_equals_Wqf32(bias_pair);

      *optr++ =
          Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), scale_vec), hf16_bias));

      bvec1 = *bptr++;  // load 32 float elements
      bvec2 = *bptr++;
      // see HVX documention for Vector shuffle and deal cross-lane
      // Q6_Vhf_equals_Wqf32 will use elements in the lower vector as the odd elements, so need to transpose here
      bias_pair = Q6_W_vdeal_VVR(Q6_Vqf32_equals_Vsf(bvec2), Q6_Vqf32_equals_Vsf(bvec1), -4);  // make a fp32 pair
      hf16_bias = Q6_Vhf_equals_Wqf32(bias_pair);

      *optr++ =
          Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), scale_vec), hf16_bias));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

    temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
    HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
    HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

    HVX_Vector bvec1 = *bptr++;  // load 32 float elements
    HVX_Vector bvec2 = *bptr++;
    HVX_VectorPair bias_pair = Q6_W_vshuff_VVR(Q6_Vqf32_equals_Vsf(bvec1), Q6_Vqf32_equals_Vsf(bvec2), -4);  // make a fp32 pair
    HVX_Vector hf16_bias = Q6_Vhf_equals_Wqf32(bias_pair);

    *optr++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), scale_vec), hf16_bias));

    bvec1 = *bptr++;  // load 32 float elements
    bvec2 = *bptr++;
    bias_pair = Q6_W_vshuff_VVR(Q6_Vqf32_equals_Vsf(bvec1), Q6_Vqf32_equals_Vsf(bvec2), -4);  // make a fp32 pair
    hf16_bias = Q6_Vhf_equals_Wqf32(bias_pair);

    *optr++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), scale_vec), hf16_bias));
  }

  return 0;
}

int32_t qhmath_hvx_dequantize_add_ui16_ahf(int8_t* restrict input, float_t* restrict bias, int8_t* restrict output,
                                           uint32_t size, float scale) {
  if ((input == NULL) || (bias == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* bptr = (HVX_Vector*)bias;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector scale_vec;

  int32_t block, l2fetch_block;
  int32_t leftover = size & 63;
  int32_t vectors_in_rounddown = size / 64;  // element number!
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  uint32_t convert = 0x80008000;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);

  scale_vec = Q6_V_vsplat_R(float_to_fp16s(scale));

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      HVX_Vector bvec1 = *bptr++;  // load 32 float elements
      HVX_Vector bvec2 = *bptr++;
      // see HVX documention for Vector shuffle and deal cross-lane
      // Q6_Vhf_equals_Wqf32 will use elements in the lower vector as the odd elements, so need to transpose here
      HVX_VectorPair bias_pair =
          Q6_W_vdeal_VVR(Q6_Vqf32_equals_Vsf(bvec2), Q6_Vqf32_equals_Vsf(bvec1), -4);  // make a fp32 pair
      HVX_Vector hf16_bias = Q6_Vhf_equals_Wqf32(bias_pair);

      HVX_Vector temp = sline1;

      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(temp, convert_vector);
      HVX_Vector qf16_val = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), scale_vec);

      *optr++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(qf16_val, hf16_bias));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    HVX_Vector bvec1 = *bptr++;  // load 32 float elements
    HVX_Vector bvec2 = *bptr++;

    // see HVX documention for Vector shuffle and deal cross-lane
    // Q6_Vhf_equals_Wqf32 will use elements in the lower vector as the odd elements, so need to transpose here
    HVX_VectorPair bias_pair = Q6_W_vdeal_VVR(Q6_Vqf32_equals_Vsf(bvec2), Q6_Vqf32_equals_Vsf(bvec1), -4);  // make a fp32 pair
    HVX_Vector hf16_bias = Q6_Vhf_equals_Wqf32(bias_pair);

    HVX_Vector temp = sline1;

    HVX_Vector sout1 = Q6_Vh_vsub_VhVh(temp, convert_vector);
    HVX_Vector qf16_val = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), scale_vec);

    *optr++ = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vadd_Vqf16Vhf(qf16_val, hf16_bias));
  }

  return 0;
}

// Only support 128x dimension
int32_t qhmath_hvx_dequantize_add_af(int8_t* restrict input, float_t* restrict bias, int8_t* restrict output, uint32_t size,
                                     float scale) {
  if ((input == NULL) || (bias == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* bptr = (HVX_Vector*)bias;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector scale_vec;
  HVX_Vector one_vec;

  int32_t block, l2fetch_block;
  int32_t leftover = size & 127;
  int32_t vectors_in_rounddown = size / 128;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  uint32_t convert = 0x00800080;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);

  scale_vec = Q6_V_vsplat_R(float_to_bits(scale));
  one_vec = Q6_V_vsplat_R(float_to_fp16s(1.0));
  HVX_Vector zero_v_sf = Q6_V_vzero();
  scale_vec = Q6_Vqf32_vadd_VsfVsf(scale_vec, Q6_V_vzero());

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
      HVX_Vector bvec1 = *bptr++;  // load 32 float elements
      HVX_Vector bvec2 = *bptr++;
      HVX_Vector bvec3 = *bptr++;
      HVX_Vector bvec4 = *bptr++;

      HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

      temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
      HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

      HVX_VectorPair result1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), one_vec);
      result1 = Q6_W_vshuff_VVR(Q6_V_hi_W(result1), Q6_V_lo_W(result1), -4);

      HVX_VectorPair result2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), one_vec);
      result2 = Q6_W_vshuff_VVR(Q6_V_hi_W(result2), Q6_V_lo_W(result2), -4);

      *optr++ = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result1), scale_vec), Q6_Vqf32_equals_Vsf(bvec1)));
      *optr++ = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result1), scale_vec), Q6_Vqf32_equals_Vsf(bvec2)));

      *optr++ = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result2), scale_vec), Q6_Vqf32_equals_Vsf(bvec3)));

      *optr++ = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result2), scale_vec), Q6_Vqf32_equals_Vsf(bvec4)));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    // NOTE: assume bias size is multiple of 128
    HVX_Vector bvec1 = *bptr++;  // load 32 float elements
    HVX_Vector bvec2 = *bptr++;
    HVX_Vector bvec3 = *bptr++;
    HVX_Vector bvec4 = *bptr++;

    HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

    temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
    HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
    HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

    HVX_VectorPair result1 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), one_vec);
    result1 = Q6_W_vshuff_VVR(Q6_V_hi_W(result1), Q6_V_lo_W(result1), -4);

    HVX_VectorPair result2 = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), one_vec);
    result2 = Q6_W_vshuff_VVR(Q6_V_hi_W(result2), Q6_V_lo_W(result2), -4);

    *optr++ = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result1), scale_vec), Q6_Vqf32_equals_Vsf(bvec1)));
    *optr++ = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result1), scale_vec), Q6_Vqf32_equals_Vsf(bvec2)));

    *optr++ = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result2), scale_vec), Q6_Vqf32_equals_Vsf(bvec3)));

    *optr++ = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result2), scale_vec), Q6_Vqf32_equals_Vsf(bvec4)));
  }

  return 0;
}

int32_t qhmath_hvx_dequantize_add_ui16_af(int8_t* restrict input, float_t* restrict bias, int8_t* restrict output,
                                          uint32_t size, float scale) {
  if ((input == NULL) || (bias == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* bptr = (HVX_Vector*)bias;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector scale_vec;
  HVX_Vector one_vec;

  int32_t block, l2fetch_block;
  int32_t leftover = size & 63;
  int32_t vectors_in_rounddown = size / 64;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  uint32_t convert = 0x80008000;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);

  scale_vec = Q6_V_vsplat_R(float_to_bits(scale));
  one_vec = Q6_V_vsplat_R(float_to_fp16s(1.0));
  scale_vec = Q6_Vqf32_vadd_VsfVsf(scale_vec, Q6_V_vzero());

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      HVX_Vector bvec1 = *bptr++;  // load 32 float elements
      HVX_Vector bvec2 = *bptr++;

      HVX_Vector temp = Q6_Vh_vsub_VhVh(sline1, convert_vector);
      HVX_VectorPair result = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(temp), one_vec);
      result = Q6_W_vshuff_VVR(Q6_V_hi_W(result), Q6_V_lo_W(result), -4);
      *optr++ = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result), scale_vec), Q6_Vqf32_equals_Vsf(bvec1)));
      *optr++ = Q6_Vsf_equals_Vqf32(
          Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result), scale_vec), Q6_Vqf32_equals_Vsf(bvec2)));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    HVX_Vector bvec1 = *bptr++;  // load 32 float elements
    HVX_Vector bvec2 = *bptr++;

    HVX_Vector temp = Q6_Vh_vsub_VhVh(sline1, convert_vector);
    HVX_VectorPair result = Q6_Wqf32_vmpy_VhfVhf(Q6_Vhf_equals_Vh(temp), one_vec);
    result = Q6_W_vshuff_VVR(Q6_V_hi_W(result), Q6_V_lo_W(result), -4);
    *optr++ = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(result), scale_vec), Q6_Vqf32_equals_Vsf(bvec1)));
    *optr++ = Q6_Vsf_equals_Vqf32(
        Q6_Vqf32_vadd_Vqf32Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(result), scale_vec), Q6_Vqf32_equals_Vsf(bvec2)));
  }

  return 0;
}

template<typename TensorType, typename TensorType1, typename TensorType2, typename TensorType3>
GraphStatus llamadequantizeaddImpl(TensorType1& out_0, const TensorType1& in_0, const TensorType& in_1,
                                   const PlainFloatTensor& scale)

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

  // HVX Method -- FP32 Version
  out_0.set_dims(in_0);

  // NHWC
  auto bias_ptr = (float*)in_1.raw_data_const();
  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  float scale_ = scale(0, 0, 0, 0);

  if (d_in % 128 != 0) { return GraphStatus::ErrorDimensions; }

  // call the kernel function for every dim() (assume total_size == bias_length)
  // NOTE: in modeling, the dequantize add can appear after linear multihead attention, so w_in * d_in == bias_length
  //      in other positions, the w_in will be 1
  if (in_0.get_dtype() == DType::QUInt8 && out_0.get_dtype() == DType::Float16) {
    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        auto in_ptr = (int8_t*)in_0.raw_data_const() + (((b * h_in) + h) * w_in * d_in);
        auto out_ptr = (int8_t*)((int16_t*)out_0.raw_data() + (((b * h_in) + h) * w_in * d_in));
        qhmath_hvx_dequantize_add_ahf(in_ptr, bias_ptr, out_ptr, w_in * d_in, scale_);
      }
    }
  } else if (in_0.get_dtype() == DType::QUInt16 && out_0.get_dtype() == DType::Float16) {
    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        auto in_ptr = (int8_t*)((int16_t*)in_0.raw_data_const() + (((b * h_in) + h) * w_in * d_in));
        auto out_ptr = (int8_t*)((int16_t*)out_0.raw_data() + (((b * h_in) + h) * w_in * d_in));
        qhmath_hvx_dequantize_add_ui16_ahf(in_ptr, bias_ptr, out_ptr, w_in * d_in, scale_);
      }
    }
  } else if (in_0.get_dtype() == DType::QUInt16 && out_0.get_dtype() == DType::Float32) {
    // NOTE: correct
    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        auto in_ptr = (int8_t*)((int16_t*)in_0.raw_data_const() + (((b * h_in) + h) * w_in * d_in));
        auto out_ptr = (int8_t*)((float_t*)out_0.raw_data() + (((b * h_in) + h) * w_in * d_in));
        qhmath_hvx_dequantize_add_ui16_af(in_ptr, bias_ptr, out_ptr, w_in * d_in, scale_);
      }
    }
  } else if (in_0.get_dtype() == DType::QUInt8 && out_0.get_dtype() == DType::Float32) {
    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        auto in_ptr = (int8_t*)in_0.raw_data_const() + (((b * h_in) + h) * w_in * d_in);
        auto out_ptr = (int8_t*)((float_t*)out_0.raw_data() + ((((b * h_in) + h) * w_in * d_in)));
        qhmath_hvx_dequantize_add_af(in_ptr, bias_ptr, out_ptr, w_in * d_in, scale_);
      }
    }
  } else {
    return GraphStatus::GraphErrorCode::ErrorUnsupported;
  }

  return GraphStatus::Success;
}
#else

template<typename TensorType, typename TensorType1, typename TensorType2, typename TensorType3>
GraphStatus llamadequantizeaddImpl(TensorType1& out_0, const TensorType1& in_0, const TensorType& in_1,
                                   const PlainFloatTensor& scale)

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

  float scale_ = scale(0, 0, 0, 0);
  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  if (in_0.get_dtype() == DType::QUInt8 && out_0.get_dtype() == DType::Float32) {
    auto out_ptr = (float*)out_0.raw_data();
    auto in_ptr = (uint8_t*)in_0.raw_data_const();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            int32_t inval = static_cast<int32_t>(*in_ptr++);
            *out_ptr++ = (inval - 128) * scale_ + in_1(0, 0, 0, w * d_in + d);
          }
        }
      }
    }
  } else if (in_0.get_dtype() == DType::QUInt16 && out_0.get_dtype() == DType::Float32) {
    auto out_ptr = (float*)out_0.raw_data();
    auto in_ptr = (uint16_t*)in_0.raw_data_const();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            int32_t inval = static_cast<int32_t>(*in_ptr++);
            *out_ptr++ = (inval - 32768) * scale_ + in_1(0, 0, 0, w * d_in + d);
          }
        }
      }
    }
  } else if (in_0.get_dtype() == DType::QUInt16 && out_0.get_dtype() == DType::Float16) {
    auto out_ptr = (__fp16*)out_0.raw_data();
    auto in_ptr = (uint16_t*)in_0.raw_data_const();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            int32_t inval = static_cast<int32_t>(*in_ptr++);
            *out_ptr++ = (__fp16)((inval - 32768) * scale_ + in_1(0, 0, 0, w * d_in + d));
          }
        }
      }
    }
  } else if (in_0.get_dtype() == DType::QUInt8 && out_0.get_dtype() == DType::Float16) {
    auto out_ptr = (__fp16*)out_0.raw_data();
    auto in_ptr = (uint8_t*)in_0.raw_data_const();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            int32_t inval = static_cast<int32_t>(*in_ptr++);
            *out_ptr++ = (__fp16)((inval - 128) * scale_ + in_1(0, 0, 0, w * d_in + d));
          }
        }
      }
    }
  }
  return GraphStatus::Success;
}

#endif

__attribute__((unused)) static float llamadequantizeaddCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_LLaMADequantizeAdd);