//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_LLaMAQuantize);

// op execute function declarations
template<typename TensorType, typename TensorType1, typename TensorType2>
GraphStatus llamaquantizeImpl(TensorType1& out_0, const TensorType1& in_0, const PlainFloatTensor& scale);

// forward declaration of sample cost function
static float llamaquantizeCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((llamaquantizeImpl<Tensor, Tensor, Tensor>), "LLaMAQuantize")
 */
DEF_PACKAGE_OP((llamaquantizeImpl<Tensor, Tensor, Tensor>), "LLaMAQuantize")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((llamaquantizeImpl<PlainFloatTensor, PlainFloatTensor, PlainFloatTensor>),
 * "LLaMAQuantize", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((llamaquantizeImpl<PlainFloatTensor, PlainFloatTensor, PlainFloatTensor>),
 * "LLaMAQuantize", llamaquantizeCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("LLaMAQuantize", "scale", true, nullptr)
#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"
#include <stddef.h>

#define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
#define L2FETCH_AHEAD (BLOCK_SIZE)

static HVX_INLINE_ALWAYS uint32_t float_to_bits(float x) {
  union {
    float f;
    uint32_t i;
  } fp32 = {.f = x};
  return fp32.i;
}

static inline int32_t float_to_fp16s(float input) {
  union {
    int32_t i;
    __fp16 f[2];
  } fp32 = {.f = {(__fp16)input, (__fp16)input}};
  return fp32.i;
}

#define FP16_MANTISA 10
#define FP16_EXPONENT_MASK 0x1f
#define FP16_EXPONENT_BIAS 0xf
#define FP16_MANTISA_MASK 0x000003ff
#define FP16_SIGN 15
#define FP16_NEG_1 0xbc00

/* execute functions for ops */
int32_t qhmath_hvx_quantize_ahf(__fp16* restrict input, __fp16* restrict output, uint32_t size, float low_level,
                                float high_level, float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec;
  int32_t block, l2fetch_block;
  // int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 64;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  HVX_Vector uintconvert = Q6_V_vsplat_R(0x80808080);

  float es = 0.5;
  low_level_vec = Q6_V_vsplat_R(float_to_fp16s(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_fp16s(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_fp16s(scale));
  es_vec = Q6_V_vsplat_R(float_to_fp16s(es));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  es_vec = Q6_Vqf16_vadd_VhfVhf(es_vec, zero_v_sf);

  HVX_Vector expmask = Q6_Vh_vsplat_R(FP16_EXPONENT_MASK);
  HVX_Vector expbias = Q6_Vh_vsplat_R(FP16_EXPONENT_BIAS);
  HVX_Vector manmask = Q6_Vh_vsplat_R(FP16_MANTISA_MASK);
  HVX_Vector exp23 = Q6_Vh_vsplat_R(23 - 1);
  HVX_Vector exp0 = Q6_Vh_vsplat_R(0 - 1);
  HVX_Vector negone = Q6_Vh_vsplat_R(FP16_NEG_1);
  HVX_Vector zero = Q6_V_vzero();

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; j += 4) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sout1 = Q6_Vqf16_vmpy_VhfVhf(sline1, scale_vec);
      sout1 = Q6_Vqf16_vadd_Vqf16Vqf16(sout1, es_vec);
      sout1 = Q6_Vhf_equals_Vqf16(sout1);
      sout1 = Q6_Vhf_vmin_VhfVhf(sout1, high_level_vec);
      sout1 = Q6_Vhf_vmax_VhfVhf(sout1, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout1, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout1, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout1, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout1, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout1, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout1, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout1, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout1, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout1 = Q6_V_vmux_QVV(expgte23, sout1, tsout1);
      }

      sout1 = Q6_Vh_equals_Vhf(sout1);

      sline2c = *iptr++;
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);

      sout2 = Q6_Vqf16_vmpy_VhfVhf(sline2, scale_vec);
      sout2 = Q6_Vqf16_vadd_Vqf16Vqf16(sout2, es_vec);
      sout2 = Q6_Vhf_equals_Vqf16(sout2);
      sout2 = Q6_Vhf_vmin_VhfVhf(sout2, high_level_vec);
      sout2 = Q6_Vhf_vmax_VhfVhf(sout2, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout2, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout2, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout2, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout2, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout2, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout2, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout2, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout2, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout2 = Q6_V_vmux_QVV(expgte23, sout2, tsout1);
      }

      sout2 = Q6_Vh_equals_Vhf(sout2);

      sline3c = *iptr++;
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);

      sout3 = Q6_Vqf16_vmpy_VhfVhf(sline3, scale_vec);
      sout3 = Q6_Vqf16_vadd_Vqf16Vqf16(sout3, es_vec);
      sout3 = Q6_Vhf_equals_Vqf16(sout3);
      sout3 = Q6_Vhf_vmin_VhfVhf(sout3, high_level_vec);
      sout3 = Q6_Vhf_vmax_VhfVhf(sout3, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout3, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout3, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout3, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout3, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout3, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout3, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout3, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout3, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout3 = Q6_V_vmux_QVV(expgte23, sout3, tsout1);
      }

      sout3 = Q6_Vh_equals_Vhf(sout3);

      sline4c = *iptr++;
      sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);

      sout4 = Q6_Vqf16_vmpy_VhfVhf(sline4, scale_vec);
      sout4 = Q6_Vqf16_vadd_Vqf16Vqf16(sout4, es_vec);
      sout4 = Q6_Vhf_equals_Vqf16(sout4);
      sout4 = Q6_Vhf_vmin_VhfVhf(sout4, high_level_vec);
      sout4 = Q6_Vhf_vmax_VhfVhf(sout4, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout4, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout4, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout4, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout4, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout4, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout4, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout4, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout4, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout4 = Q6_V_vmux_QVV(expgte23, sout4, tsout1);
      }

      sout4 = Q6_Vh_equals_Vhf(sout4);

      HVX_Vector reql_h = Q6_Vb_vpack_VhVh_sat(sout2, sout1);
      *optr++ = Q6_Vb_vadd_VbVb(reql_h, uintconvert);

      HVX_Vector reqh_h = Q6_Vb_vpack_VhVh_sat(sout4, sout3);
      *optr++ = Q6_Vb_vadd_VbVb(reqh_h, uintconvert);

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;
    }
  }

  return 0;
}

int32_t qhmath_hvx_quantize_ui16_ahf(__fp16* restrict input, __fp16* restrict output, uint32_t size, float low_level,
                                     float high_level, float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec;
  int32_t block, l2fetch_block;
  // int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 64;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  HVX_Vector uintconvert = Q6_V_vsplat_R(0x80008000);

  float es = 0.5;
  low_level_vec = Q6_V_vsplat_R(float_to_fp16s(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_fp16s(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_fp16s(scale));
  es_vec = Q6_V_vsplat_R(float_to_fp16s(es));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  es_vec = Q6_Vqf16_vadd_VhfVhf(es_vec, zero_v_sf);

  HVX_Vector expmask = Q6_Vh_vsplat_R(FP16_EXPONENT_MASK);
  HVX_Vector expbias = Q6_Vh_vsplat_R(FP16_EXPONENT_BIAS);
  HVX_Vector manmask = Q6_Vh_vsplat_R(FP16_MANTISA_MASK);
  HVX_Vector exp23 = Q6_Vh_vsplat_R(23 - 1);
  HVX_Vector exp0 = Q6_Vh_vsplat_R(0 - 1);
  HVX_Vector negone = Q6_Vh_vsplat_R(FP16_NEG_1);
  HVX_Vector zero = Q6_V_vzero();

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; j += 4) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sout1 = Q6_Vqf16_vmpy_VhfVhf(sline1, scale_vec);
      sout1 = Q6_Vqf16_vadd_Vqf16Vqf16(sout1, es_vec);
      sout1 = Q6_Vhf_equals_Vqf16(sout1);
      sout1 = Q6_Vhf_vmin_VhfVhf(sout1, high_level_vec);
      sout1 = Q6_Vhf_vmax_VhfVhf(sout1, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout1, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout1, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout1, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout1, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout1, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout1, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout1, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout1, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout1 = Q6_V_vmux_QVV(expgte23, sout1, tsout1);
      }

      sout1 = Q6_Vh_equals_Vhf(sout1);

      sline2c = *iptr++;
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);

      sout2 = Q6_Vqf16_vmpy_VhfVhf(sline2, scale_vec);
      sout2 = Q6_Vqf16_vadd_Vqf16Vqf16(sout2, es_vec);
      sout2 = Q6_Vhf_equals_Vqf16(sout2);
      sout2 = Q6_Vhf_vmin_VhfVhf(sout2, high_level_vec);
      sout2 = Q6_Vhf_vmax_VhfVhf(sout2, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout2, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout2, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout2, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout2, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout2, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout2, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout2, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout2, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout2 = Q6_V_vmux_QVV(expgte23, sout2, tsout1);
      }

      sout2 = Q6_Vh_equals_Vhf(sout2);

      sline3c = *iptr++;
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);

      sout3 = Q6_Vqf16_vmpy_VhfVhf(sline3, scale_vec);
      sout3 = Q6_Vqf16_vadd_Vqf16Vqf16(sout3, es_vec);
      sout3 = Q6_Vhf_equals_Vqf16(sout3);
      sout3 = Q6_Vhf_vmin_VhfVhf(sout3, high_level_vec);
      sout3 = Q6_Vhf_vmax_VhfVhf(sout3, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout3, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout3, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout3, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout3, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout3, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout3, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout3, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout3, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout3 = Q6_V_vmux_QVV(expgte23, sout3, tsout1);
      }

      sout3 = Q6_Vh_equals_Vhf(sout3);

      sline4c = *iptr++;
      sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);

      sout4 = Q6_Vqf16_vmpy_VhfVhf(sline4, scale_vec);
      sout4 = Q6_Vqf16_vadd_Vqf16Vqf16(sout4, es_vec);
      sout4 = Q6_Vhf_equals_Vqf16(sout4);
      sout4 = Q6_Vhf_vmin_VhfVhf(sout4, high_level_vec);
      sout4 = Q6_Vhf_vmax_VhfVhf(sout4, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout4, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout4, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout4, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout4, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout4, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout4, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout4, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout4, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout4 = Q6_V_vmux_QVV(expgte23, sout4, tsout1);
      }

      sout4 = Q6_Vh_equals_Vhf(sout4);

      *optr++ = Q6_Vh_vadd_VhVh(sout1, uintconvert);
      *optr++ = Q6_Vh_vadd_VhVh(sout2, uintconvert);
      *optr++ = Q6_Vh_vadd_VhVh(sout3, uintconvert);
      *optr++ = Q6_Vh_vadd_VhVh(sout4, uintconvert);

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;
    }
  }

  return 0;
}

int32_t qhmath_hvx_quantize_ahf_int8(__fp16* restrict input, __fp16* restrict output, uint32_t size, float low_level,
                                     float high_level, float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec;
  int32_t block, l2fetch_block;
  // int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 64;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  float es = 0.5;
  low_level_vec = Q6_V_vsplat_R(float_to_fp16s(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_fp16s(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_fp16s(scale));
  es_vec = Q6_V_vsplat_R(float_to_fp16s(es));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  es_vec = Q6_Vqf16_vadd_VhfVhf(es_vec, zero_v_sf);

  HVX_Vector expmask = Q6_Vh_vsplat_R(FP16_EXPONENT_MASK);
  HVX_Vector expbias = Q6_Vh_vsplat_R(FP16_EXPONENT_BIAS);
  HVX_Vector manmask = Q6_Vh_vsplat_R(FP16_MANTISA_MASK);
  HVX_Vector exp23 = Q6_Vh_vsplat_R(23 - 1);
  HVX_Vector exp0 = Q6_Vh_vsplat_R(0 - 1);
  HVX_Vector negone = Q6_Vh_vsplat_R(FP16_NEG_1);
  HVX_Vector zero = Q6_V_vzero();

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; j += 4) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sout1 = Q6_Vqf16_vmpy_VhfVhf(sline1, scale_vec);
      sout1 = Q6_Vqf16_vadd_Vqf16Vqf16(sout1, es_vec);
      sout1 = Q6_Vhf_equals_Vqf16(sout1);
      sout1 = Q6_Vhf_vmin_VhfVhf(sout1, high_level_vec);
      sout1 = Q6_Vhf_vmax_VhfVhf(sout1, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout1, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout1, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout1, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout1, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout1, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout1, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout1, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout1, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout1 = Q6_V_vmux_QVV(expgte23, sout1, tsout1);
      }

      sout1 = Q6_Vh_equals_Vhf(sout1);

      sline2c = *iptr++;
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);

      sout2 = Q6_Vqf16_vmpy_VhfVhf(sline2, scale_vec);
      sout2 = Q6_Vqf16_vadd_Vqf16Vqf16(sout2, es_vec);
      sout2 = Q6_Vhf_equals_Vqf16(sout2);
      sout2 = Q6_Vhf_vmin_VhfVhf(sout2, high_level_vec);
      sout2 = Q6_Vhf_vmax_VhfVhf(sout2, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout2, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout2, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout2, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout2, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout2, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout2, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout2, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout2, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout2 = Q6_V_vmux_QVV(expgte23, sout2, tsout1);
      }

      sout2 = Q6_Vh_equals_Vhf(sout2);

      sline3c = *iptr++;
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);

      sout3 = Q6_Vqf16_vmpy_VhfVhf(sline3, scale_vec);
      sout3 = Q6_Vqf16_vadd_Vqf16Vqf16(sout3, es_vec);
      sout3 = Q6_Vhf_equals_Vqf16(sout3);
      sout3 = Q6_Vhf_vmin_VhfVhf(sout3, high_level_vec);
      sout3 = Q6_Vhf_vmax_VhfVhf(sout3, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout3, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout3, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout3, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout3, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout3, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout3, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout3, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout3, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout3 = Q6_V_vmux_QVV(expgte23, sout3, tsout1);
      }

      sout3 = Q6_Vh_equals_Vhf(sout3);

      sline4c = *iptr++;
      sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);

      sout4 = Q6_Vqf32_vmpy_VsfVsf(sline4, scale_vec);
      sout4 = Q6_Vqf16_vadd_Vqf16Vqf16(sout4, es_vec);
      sout4 = Q6_Vhf_equals_Vqf16(sout4);
      sout4 = Q6_Vhf_vmin_VhfVhf(sout4, high_level_vec);
      sout4 = Q6_Vhf_vmax_VhfVhf(sout4, low_level_vec);

      {
        HVX_Vector exp = Q6_Vh_vasr_VhR(sout4, FP16_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vh_vsub_VhVh(exp, expbias);

        HVX_Vector man = Q6_Vh_vasr_VhVh(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout4, man);

        HVX_Vector sign = Q6_Vh_vasr_VhR(sout4, FP16_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VhVh(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VhVh(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VhVh(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VhVh(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vh_vadd_VhVh(sout4, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout4, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vh_vasl_VhR(sout4, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VhVh(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout4, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout4, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout4 = Q6_V_vmux_QVV(expgte23, sout4, tsout1);
      }

      sout4 = Q6_Vh_equals_Vhf(sout4);

      HVX_Vector reql_h = Q6_Vb_vpack_VhVh_sat(sout2, sout1);
      *optr++ = reql_h;

      HVX_Vector reqh_h = Q6_Vb_vpack_VhVh_sat(sout4, sout3);
      *optr++ = reqh_h;

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;
    }
  }

  return 0;
}

#define FLOAT_MANTISA 23
#define FLOAT_EXPONENT_MASK 0xff
#define FLOAT_EXPONENT_BIAS 0x7f
#define FLOAT_MANTISA_MASK 0x007fffff
#define FLOAT_SIGN 31
#define FLOAT_NEG_1 0xBF800000
#define ROUND_2_SCALE 22
#define ROUND_SCALSE ((1 << ROUND_2_SCALE) * 1.0f)

int32_t qhmath_hvx_quantize_af(float* restrict input, float* restrict output, uint32_t size, float low_level, float high_level,
                               float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec, round_scale_vec;
  int32_t block, l2fetch_block;
  // int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  float es = 0.5f;
  low_level_vec = Q6_V_vsplat_R(float_to_bits(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_bits(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_bits(scale));
  es_vec = Q6_V_vsplat_R(float_to_bits(es));
  round_scale_vec = Q6_V_vsplat_R(float_to_bits(ROUND_SCALSE));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  es_vec = Q6_Vqf32_vadd_VsfVsf(es_vec, zero_v_sf);

  HVX_Vector uintconvert = Q6_V_vsplat_R(0x80808080);

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; j += 4) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sout1 = Q6_Vqf32_vmpy_VsfVsf(sline1, scale_vec);
      sout1 = Q6_Vqf32_vadd_Vqf32Vqf32(sout1, es_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);
      sout1 = Q6_Vsf_vmin_VsfVsf(sout1, high_level_vec);
      sout1 = Q6_Vsf_vmax_VsfVsf(sout1, low_level_vec);
      sout1 = Q6_Vqf32_vmpy_VsfVsf(sout1, round_scale_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);

      sout1 = Q6_Vw_equals_Vsf(sout1);
      sout1 = Q6_Vw_vasr_VwR(sout1, ROUND_2_SCALE);
      // sout1 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout1, Q6_V_vzero()), 0);

      sline2c = *iptr++;
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);

      sout2 = Q6_Vqf32_vmpy_VsfVsf(sline2, scale_vec);
      sout2 = Q6_Vqf32_vadd_Vqf32Vqf32(sout2, es_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);
      sout2 = Q6_Vsf_vmin_VsfVsf(sout2, high_level_vec);
      sout2 = Q6_Vsf_vmax_VsfVsf(sout2, low_level_vec);
      sout2 = Q6_Vqf32_vmpy_VsfVsf(sout2, round_scale_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);

      sout2 = Q6_Vw_equals_Vsf(sout2);
      sout2 = Q6_Vw_vasr_VwR(sout2, ROUND_2_SCALE);
      // sout2 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout2, Q6_V_vzero()), 0);

      sline3c = *iptr++;
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);

      sout3 = Q6_Vqf32_vmpy_VsfVsf(sline3, scale_vec);
      sout3 = Q6_Vqf32_vadd_Vqf32Vqf32(sout3, es_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);
      sout3 = Q6_Vsf_vmin_VsfVsf(sout3, high_level_vec);
      sout3 = Q6_Vsf_vmax_VsfVsf(sout3, low_level_vec);
      sout3 = Q6_Vqf32_vmpy_VsfVsf(sout3, round_scale_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);

      sout3 = Q6_Vw_equals_Vsf(sout3);
      sout3 = Q6_Vw_vasr_VwR(sout3, ROUND_2_SCALE);
      // sout3 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout3, Q6_V_vzero()), 0);

      sline4c = *iptr++;
      sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);

      sout4 = Q6_Vqf32_vmpy_VsfVsf(sline4, scale_vec);
      sout4 = Q6_Vqf32_vadd_Vqf32Vqf32(sout4, es_vec);
      sout4 = Q6_Vsf_equals_Vqf32(sout4);
      sout4 = Q6_Vsf_vmin_VsfVsf(sout4, high_level_vec);
      sout4 = Q6_Vsf_vmax_VsfVsf(sout4, low_level_vec);
      sout4 = Q6_Vqf32_vmpy_VsfVsf(sout4, round_scale_vec);
      sout4 = Q6_Vsf_equals_Vqf32(sout4);

      sout4 = Q6_Vw_equals_Vsf(sout4);
      sout4 = Q6_Vw_vasr_VwR(sout4, ROUND_2_SCALE);
      // sout4 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout4, Q6_V_vzero()), 0);

      HVX_Vector reql_h = Q6_Vh_vpack_VwVw_sat(sout2, sout1);
      HVX_Vector reqh_h = Q6_Vh_vpack_VwVw_sat(sout4, sout3);
      HVX_Vector req_b = Q6_Vb_vpack_VhVh_sat(reqh_h, reql_h);

      *optr++ = Q6_Vb_vadd_VbVb(req_b, uintconvert);

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;
    }
  }

  return 0;
}

#define INT16_ROUND_2_SCALE 15
#define INT16_ROUND_SCALSE ((1 << INT16_ROUND_2_SCALE) * 1.0f)

int32_t qhmath_hvx_quantize_ui16_af(float* restrict input, float* restrict output, uint32_t size, float low_level,
                                    float high_level, float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec, round_scale_vec;
  int32_t block, l2fetch_block;
  // int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  float es = 0.5f;
  low_level_vec = Q6_V_vsplat_R(float_to_bits(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_bits(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_bits(scale));
  es_vec = Q6_V_vsplat_R(float_to_bits(es));
  round_scale_vec = Q6_V_vsplat_R(float_to_bits(INT16_ROUND_SCALSE));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  es_vec = Q6_Vqf32_vadd_VsfVsf(es_vec, zero_v_sf);

  HVX_Vector uintconvert = Q6_V_vsplat_R(0x80008000);

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; j += 4) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sout1 = Q6_Vqf32_vmpy_VsfVsf(sline1, scale_vec);
      sout1 = Q6_Vqf32_vadd_Vqf32Vqf32(sout1, es_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);
      sout1 = Q6_Vsf_vmin_VsfVsf(sout1, high_level_vec);
      sout1 = Q6_Vsf_vmax_VsfVsf(sout1, low_level_vec);
      sout1 = Q6_Vqf32_vmpy_VsfVsf(sout1, round_scale_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);

      sout1 = Q6_Vw_equals_Vsf(sout1);
      sout1 = Q6_Vw_vasr_VwR(sout1, INT16_ROUND_2_SCALE);
      // sout1 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout1, Q6_V_vzero()), 0);

      sline2c = *iptr++;
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);

      sout2 = Q6_Vqf32_vmpy_VsfVsf(sline2, scale_vec);
      sout2 = Q6_Vqf32_vadd_Vqf32Vqf32(sout2, es_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);
      sout2 = Q6_Vsf_vmin_VsfVsf(sout2, high_level_vec);
      sout2 = Q6_Vsf_vmax_VsfVsf(sout2, low_level_vec);
      sout2 = Q6_Vqf32_vmpy_VsfVsf(sout2, round_scale_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);

      sout2 = Q6_Vw_equals_Vsf(sout2);
      sout2 = Q6_Vw_vasr_VwR(sout2, INT16_ROUND_2_SCALE);
      // sout2 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout2, Q6_V_vzero()), 0);

      sline3c = *iptr++;
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);

      sout3 = Q6_Vqf32_vmpy_VsfVsf(sline3, scale_vec);
      sout3 = Q6_Vqf32_vadd_Vqf32Vqf32(sout3, es_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);
      sout3 = Q6_Vsf_vmin_VsfVsf(sout3, high_level_vec);
      sout3 = Q6_Vsf_vmax_VsfVsf(sout3, low_level_vec);
      sout3 = Q6_Vqf32_vmpy_VsfVsf(sout3, round_scale_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);

      sout3 = Q6_Vw_equals_Vsf(sout3);
      sout3 = Q6_Vw_vasr_VwR(sout3, INT16_ROUND_2_SCALE);
      // sout3 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout3, Q6_V_vzero()), 0);

      sline4c = *iptr++;
      sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);

      sout4 = Q6_Vqf32_vmpy_VsfVsf(sline4, scale_vec);
      sout4 = Q6_Vqf32_vadd_Vqf32Vqf32(sout4, es_vec);
      sout4 = Q6_Vsf_equals_Vqf32(sout4);
      sout4 = Q6_Vsf_vmin_VsfVsf(sout4, high_level_vec);
      sout4 = Q6_Vsf_vmax_VsfVsf(sout4, low_level_vec);
      sout4 = Q6_Vqf32_vmpy_VsfVsf(sout4, round_scale_vec);
      sout4 = Q6_Vsf_equals_Vqf32(sout4);

      sout4 = Q6_Vw_equals_Vsf(sout4);
      sout4 = Q6_Vw_vasr_VwR(sout4, INT16_ROUND_2_SCALE);
      // sout4 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout4, Q6_V_vzero()), 0);

      HVX_Vector reql_h = Q6_Vh_vpack_VwVw_sat(sout2, sout1);
      HVX_Vector reqh_h = Q6_Vh_vpack_VwVw_sat(sout4, sout3);

      *optr++ = Q6_Vh_vadd_VhVh(reql_h, uintconvert);
      *optr++ = Q6_Vh_vadd_VhVh(reqh_h, uintconvert);

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;
    }
  }

  return 0;
}

int32_t qhmath_hvx_quantize_af_out_int8(float* restrict input, float* restrict output, uint32_t size, float low_level,
                                        float high_level, float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_UVector* optr = (HVX_UVector*)output;

  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec;
  int32_t block, l2fetch_block;
  // int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  float es = 0.5f;
  low_level_vec = Q6_V_vsplat_R(float_to_bits(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_bits(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_bits(scale));
  es_vec = Q6_V_vsplat_R(float_to_bits(es));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  es_vec = Q6_Vqf32_vadd_VsfVsf(es_vec, zero_v_sf);

  HVX_Vector expmask = Q6_V_vsplat_R(FLOAT_EXPONENT_MASK);
  HVX_Vector expbias = Q6_V_vsplat_R(FLOAT_EXPONENT_BIAS);
  HVX_Vector manmask = Q6_V_vsplat_R(FLOAT_MANTISA_MASK);
  HVX_Vector exp23 = Q6_V_vsplat_R(23 - 1);
  HVX_Vector exp0 = Q6_V_vsplat_R(0 - 1);
  HVX_Vector negone = Q6_V_vsplat_R(FLOAT_NEG_1);
  HVX_Vector zero = Q6_V_vzero();

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; j += 4) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sout1 = Q6_Vqf32_vmpy_VsfVsf(sline1, scale_vec);
      sout1 = Q6_Vqf32_vadd_Vqf32Vqf32(sout1, es_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);
      sout1 = Q6_Vsf_vmin_VsfVsf(sout1, high_level_vec);
      sout1 = Q6_Vsf_vmax_VsfVsf(sout1, low_level_vec);

      {
        HVX_Vector exp = Q6_Vw_vasr_VwR(sout1, FLOAT_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vw_vsub_VwVw(exp, expbias);

        HVX_Vector man = Q6_Vw_vasr_VwVw(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout1, man);

        HVX_Vector sign = Q6_Vw_vasr_VwR(sout1, FLOAT_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VwVw(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VwVw(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VwVw(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VwVw(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vw_vadd_VwVw(sout1, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout1, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vw_vasl_VwR(sout1, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VwVw(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout1, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout1, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout1 = Q6_V_vmux_QVV(expgte23, sout1, tsout1);
      }

      sout1 = Q6_Vw_equals_Vsf(sout1);
      // sout1 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout1, Q6_V_vzero()), 0);

      sline2c = *iptr++;
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);

      sout2 = Q6_Vqf32_vmpy_VsfVsf(sline2, scale_vec);
      sout2 = Q6_Vqf32_vadd_Vqf32Vqf32(sout2, es_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);
      sout2 = Q6_Vsf_vmin_VsfVsf(sout2, high_level_vec);
      sout2 = Q6_Vsf_vmax_VsfVsf(sout2, low_level_vec);

      {
        HVX_Vector exp = Q6_Vw_vasr_VwR(sout2, FLOAT_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vw_vsub_VwVw(exp, expbias);

        HVX_Vector man = Q6_Vw_vasr_VwVw(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout2, man);

        HVX_Vector sign = Q6_Vw_vasr_VwR(sout2, FLOAT_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VwVw(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VwVw(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VwVw(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VwVw(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vw_vadd_VwVw(sout2, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout2, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vw_vasl_VwR(sout2, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VwVw(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout2, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout2, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout2 = Q6_V_vmux_QVV(expgte23, sout2, tsout1);
      }

      sout2 = Q6_Vw_equals_Vsf(sout2);
      // sout2 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout2, Q6_V_vzero()), 0);

      sline3c = *iptr++;
      sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);

      sout3 = Q6_Vqf32_vmpy_VsfVsf(sline3, scale_vec);
      sout3 = Q6_Vqf32_vadd_Vqf32Vqf32(sout3, es_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);
      sout3 = Q6_Vsf_vmin_VsfVsf(sout3, high_level_vec);
      sout3 = Q6_Vsf_vmax_VsfVsf(sout3, low_level_vec);

      {
        HVX_Vector exp = Q6_Vw_vasr_VwR(sout3, FLOAT_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vw_vsub_VwVw(exp, expbias);

        HVX_Vector man = Q6_Vw_vasr_VwVw(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout3, man);

        HVX_Vector sign = Q6_Vw_vasr_VwR(sout3, FLOAT_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VwVw(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VwVw(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VwVw(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VwVw(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vw_vadd_VwVw(sout3, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout3, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vw_vasl_VwR(sout3, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VwVw(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout3, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout3, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout3 = Q6_V_vmux_QVV(expgte23, sout3, tsout1);
      }

      sout3 = Q6_Vw_equals_Vsf(sout3);
      // sout3 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout3, Q6_V_vzero()), 0);

      sline4c = *iptr++;
      sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);

      sout4 = Q6_Vqf32_vmpy_VsfVsf(sline4, scale_vec);
      sout4 = Q6_Vqf32_vadd_Vqf32Vqf32(sout4, es_vec);
      sout4 = Q6_Vsf_equals_Vqf32(sout4);
      sout4 = Q6_Vsf_vmin_VsfVsf(sout4, high_level_vec);
      sout4 = Q6_Vsf_vmax_VsfVsf(sout4, low_level_vec);

      {
        HVX_Vector exp = Q6_Vw_vasr_VwR(sout4, FLOAT_MANTISA);
        exp = Q6_V_vand_VV(exp, expmask);
        exp = Q6_Vw_vsub_VwVw(exp, expbias);

        HVX_Vector man = Q6_Vw_vasr_VwVw(manmask, exp);
        HVX_Vector manzero = Q6_V_vand_VV(sout4, man);

        HVX_Vector sign = Q6_Vw_vasr_VwR(sout4, FLOAT_SIGN);
        HVX_Vector issignpos = Q6_Q_vcmp_eq_VwVw(sign, zero);

        HVX_Vector expgte23 = Q6_Q_vcmp_gt_VwVw(exp, exp23);
        HVX_Vector expgte0 = Q6_Q_vcmp_gt_VwVw(exp, exp0);
        HVX_Vector maneqzero = Q6_Q_vcmp_eq_VwVw(manzero, zero);

        HVX_Vector exppos_signneg = Q6_Vw_vadd_VwVw(sout4, man);
        man = Q6_V_vnot_V(man);
        HVX_Vector exppos_signpos = Q6_V_vand_VV(sout4, man);
        exppos_signneg = Q6_V_vand_VV(exppos_signneg, man);
        HVX_Vector shift1 = Q6_Vw_vasl_VwR(sout4, 1);
        HVX_Vector iszero = Q6_Q_vcmp_eq_VwVw(shift1, zero);

        // exp >= 0
        HVX_Vector tsout1 = Q6_V_vmux_QVV(issignpos, exppos_signpos, exppos_signneg);
        tsout1 = Q6_V_vmux_QVV(maneqzero, sout4, tsout1);

        // exp < 0 (-1, 1)
        HVX_Vector tsout2 = Q6_V_vmux_QVV(iszero, sout4, negone);
        tsout2 = Q6_V_vmux_QVV(issignpos, zero, tsout2);

        tsout1 = Q6_V_vmux_QVV(expgte0, tsout1, tsout2);
        sout4 = Q6_V_vmux_QVV(expgte23, sout4, tsout1);
      }

      sout4 = Q6_Vw_equals_Vsf(sout4);
      // sout4 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout4, Q6_V_vzero()), 0);

      HVX_Vector reql_h = Q6_Vh_vpack_VwVw_sat(sout2, sout1);
      HVX_Vector reqh_h = Q6_Vh_vpack_VwVw_sat(sout4, sout3);
      HVX_Vector req_b = Q6_Vb_vpack_VhVh_sat(reqh_h, reql_h);

      *optr++ = req_b;

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;
    }
  }

  return 0;
}

template<typename TensorType, typename TensorType1, typename TensorType2>
GraphStatus llamaquantizeImpl(TensorType1& out_0, const TensorType1& in_0, const PlainFloatTensor& scale)

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
  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  float scale_ = scale(0, 0, 0, 0);

  scale_ = 1.0f / scale_;

  size_t size = b_in * h_in * w_in * d_in;
  DType dtype = in_0.get_dtype();

  if (dtype == DType::Float16 && out_0.get_dtype() == DType::QUInt8) {
    // NHWC
    auto in_ptr = (__fp16*)in_0.raw_data_const();
    auto out_ptr = (__fp16*)out_0.raw_data();

    qhmath_hvx_quantize_ahf(in_ptr, out_ptr, size, -128.0f, 127.0f, scale_);
  }
  if (dtype == DType::Float32 && out_0.get_dtype() == DType::QUInt8) {
    // NHWC
    auto in_ptr = (float*)in_0.raw_data_const();
    auto out_ptr = (float*)out_0.raw_data();
    qhmath_hvx_quantize_af(in_ptr, out_ptr, size, -128.0f, 127.0f, scale_);
  }

  if (dtype == DType::Float16 && out_0.get_dtype() == DType::QUInt16) {
    // NHWC
    auto in_ptr = (__fp16*)in_0.raw_data_const();
    auto out_ptr = (__fp16*)out_0.raw_data();

    qhmath_hvx_quantize_ui16_ahf(in_ptr, out_ptr, size, -32768.0f, 32767.0f, scale_);
  }
  if (dtype == DType::Float32 && out_0.get_dtype() == DType::QUInt16) {
    // NHWC
    auto in_ptr = (float*)in_0.raw_data_const();
    auto out_ptr = (float*)out_0.raw_data();
    qhmath_hvx_quantize_ui16_af(in_ptr, out_ptr, size, -32768.0f, 32767.0f, scale_);
  }

  if (dtype == DType::Float16 && out_0.get_dtype() == DType::QInt8) {
    // NHWC
    auto in_ptr = (__fp16*)in_0.raw_data_const();
    auto out_ptr = (__fp16*)out_0.raw_data();

    qhmath_hvx_quantize_ahf_int8(in_ptr, out_ptr, size, -128.0f, 127.0f, scale_);
  }

  if (dtype == DType::Float32 && out_0.get_dtype() == DType::QInt8) {
    // NHWC
    auto in_ptr = (float*)in_0.raw_data_const();
    auto out_ptr = (float*)out_0.raw_data();
    qhmath_hvx_quantize_af_out_int8(in_ptr, out_ptr, size, -128.0f, 127.0f, scale_);
  }

  //   auto out_ptr = (int8_t*)out_0.raw_data();

  //   out_ptr[0] = (int)dtype;
  //   out_ptr[1] = (int)out_0.get_dtype();

  return GraphStatus::Success;
}

#else

extern float Round(float num);

template<typename TensorType, typename TensorType1, typename TensorType2>
GraphStatus llamaquantizeImpl(TensorType1& out_0, const TensorType1& in_0, const PlainFloatTensor& scale)

{
  out_0.set_dims(in_0);

  float scale_ = scale(0, 0, 0, 0);

  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  DType dtype = in_0.get_dtype();
  if (dtype == DType::Float32 && out_0.get_dtype() == DType::QUInt8) {
    auto out_ptr = (int8_t*)out_0.raw_data();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            float inval = in_0(b, h, w, d);

            // float result = Round(inval / scale_);

            long v = lroundf(inval / scale_);

            if (v > 127) v = 127;

            if (v < -128) v = -128;

            v += 128;

            *out_ptr++ = static_cast<uint8_t>(v);
          }
        }
      }
    }
  }

  if (dtype == DType::Float32 && out_0.get_dtype() == DType::QUInt16) {
    auto out_ptr = (int16_t*)out_0.raw_data();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            float inval = in_0(b, h, w, d);

            // float result = Round(inval / scale_);

            long v = lroundf(inval / scale_);

            if (v > 32767) v = 32767;

            if (v < -32768) v = -32768;

            v += 32768;

            *out_ptr++ = static_cast<uint16_t>(v);
          }
        }
      }
    }
  }

  if (dtype == DType::Float32 && out_0.get_dtype() == DType::QUInt8) {
    auto out_ptr = (int8_t*)out_0.raw_data();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            float inval = in_0(b, h, w, d);

            // float result = Round(inval / scale_);

            long v = lroundf(inval / scale_);

            if (v > 127) v = 127;

            if (v < -128) v = -128;

            v += 128;

            *out_ptr++ = static_cast<uint8_t>(v);
          }
        }
      }
    }
  }

  if (dtype == DType::Float16 && out_0.get_dtype() == DType::QUInt16) {
    auto out_ptr = (int16_t*)out_0.raw_data();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          for (Idx d = 0; d < d_in; d++) {
            float inval = in_0(b, h, w, d);

            // float result = Round(inval / scale_);

            long v = lroundf(inval / scale_);

            if (v > 32767) v = 32767;

            if (v < -32768) v = -32768;

            v += 32768;

            *out_ptr++ = static_cast<uint16_t>(v);
          }
        }
      }
    }
  }

  return GraphStatus::Success;
}
#endif

__attribute__((unused)) static float llamaquantizeCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_LLaMAQuantize);