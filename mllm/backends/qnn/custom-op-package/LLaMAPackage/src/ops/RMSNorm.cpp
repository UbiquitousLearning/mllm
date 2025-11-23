//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_RMSNorm);

// op execute function declarations
template<typename TensorType>
GraphStatus rmsnormImpl(TensorType& out_0, const TensorType& in_0, const TensorType& weights);

// forward declaration of sample cost function
static float rmsnormCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((rmsnormImpl<Tensor>), "RMSNorm")
 */
DEF_PACKAGE_OP((rmsnormImpl<Tensor>), "RMSNorm")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((rmsnormImpl<PlainFloatTensor>), "RMSNorm", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((rmsnormImpl<PlainFloatTensor>),
 * "RMSNorm", rmsnormCostFunc, Flags::RESOURCE_HVX)
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

int32_t hvx_rmsnorm_af(float* restrict input, float* restrict weights, float* restrict output, uint32_t size) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)weights;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;

  int32_t block, l2fetch_block;
  int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  // ^2 sum
  HVX_Vector sum = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));
  }

  float epsilon_ = 1e-6;
  union {
    float f;
    uint32_t ui;
  } sum_value;
  sum_value.f = 0.0f;

  HVX_Vector zero = Q6_V_vzero();

  for (int32_t i = 64; i >= 4; i >>= 1) { sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vlalign_VVR(sum, zero, i)); }

  sum = Q6_Vsf_equals_Vqf32(sum);
  sum_value.f = 1.0f / sqrtf(*((float*)&sum + 31) / size + epsilon_);

  // x * 1/rsqrt(sum)
  iptr = (HVX_Vector*)input;
  sline1p = *iptr++;
  sline2p = *iptr2++;

  HVX_Vector irsqrt_vsf = Q6_V_vsplat_R(sum_value.ui);
  HVX_Vector irsqrt_vqf32 = Q6_Vqf32_vadd_VsfVsf(irsqrt_vsf, Q6_V_vzero());

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
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)weights);

      HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1, sline2);
      *optr++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32));

      sline1p = sline1c;
      sline2p = sline2c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    sline2c = is_aligned(iptr2, VLEN) && leftover == 0 ? sline2p : *iptr2++;
    sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)weights);

    HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1, sline2);
    *optr++ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32));
  }

  if (leftover_size > 0) return -1;

  return 0;
}

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

#define FLOAT_MANTISA 23
#define FLOAT_EXPONENT_MASK 0xff
#define FLOAT_EXPONENT_BIAS 0x7f
#define FLOAT_MANTISA_MASK 0x007fffff
#define FLOAT_SIGN 31
#define FLOAT_NEG_1 0xBF800000
#define ROUND_2_SCALE 22
#define ROUND_SCALSE ((1 << ROUND_2_SCALE) * 1.0f)

int32_t hvx_rmsnorm_auint8(float* restrict input, float* restrict weights, uint8_t* restrict output, uint32_t size,
                           float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)weights;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;
  HVX_Vector slinewp, slinewc, slinew;

  HVX_Vector sout1, sout2, sout3, sout4;
  HVX_Vector low_level_vec, high_level_vec, scale_vec, es_vec, round_scale_vec;

  float low_level = -128.0f;
  float high_level = 127.0f;

  float es = 0.5f;
  low_level_vec = Q6_V_vsplat_R(float_to_bits(low_level));
  high_level_vec = Q6_V_vsplat_R(float_to_bits(high_level));
  scale_vec = Q6_V_vsplat_R(float_to_bits(scale));
  es_vec = Q6_V_vsplat_R(float_to_bits(es));
  round_scale_vec = Q6_V_vsplat_R(float_to_bits(ROUND_SCALSE));

  HVX_Vector zero_v_sf = Q6_V_vzero();
  scale_vec = Q6_Vqf32_vadd_VsfVsf(scale_vec, zero_v_sf);
  es_vec = Q6_Vqf32_vadd_VsfVsf(es_vec, zero_v_sf);

  HVX_Vector uintconvert = Q6_V_vsplat_R(0x80808080);

  HVX_Vector zero = Q6_V_vzero();

  int32_t block, l2fetch_block;
  int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  // ^2 sum
  HVX_Vector sum = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));
  }

  float epsilon_ = 1e-6;
  union {
    float f;
    uint32_t ui;
  } sum_value;
  sum_value.f = 0.0f;

  for (int32_t i = 64; i >= 4; i >>= 1) { sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vlalign_VVR(sum, zero, i)); }

  sum = Q6_Vsf_equals_Vqf32(sum);
  sum_value.f = 1.0f / sqrtf(*((float*)&sum + 31) / size + epsilon_);

  // x * 1/rsqrt(sum)
  iptr = (HVX_Vector*)input;

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  slinewp = *iptr2++;

  HVX_Vector irsqrt_vsf = Q6_V_vsplat_R(sum_value.ui);
  HVX_Vector irsqrt_vqf32 = Q6_Vqf32_vadd_VsfVsf(irsqrt_vsf, Q6_V_vzero());

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t j = 0; j < block; j += 4) {
      {
        sline1c = *iptr++;
        slinewc = *iptr2++;
        sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1, slinew);
        sline1 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      sout1 = Q6_Vqf32_vmpy_Vqf32Vqf32(sline1, scale_vec);
      sout1 = Q6_Vqf32_vadd_Vqf32Vqf32(sout1, es_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);
      sout1 = Q6_Vsf_vmin_VsfVsf(sout1, high_level_vec);
      sout1 = Q6_Vsf_vmax_VsfVsf(sout1, low_level_vec);
      sout1 = Q6_Vqf32_vmpy_VsfVsf(sout1, round_scale_vec);
      sout1 = Q6_Vsf_equals_Vqf32(sout1);

      sout1 = Q6_Vw_equals_Vsf(sout1);
      sout1 = Q6_Vw_vasr_VwR(sout1, ROUND_2_SCALE);
      // sout1 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout1, Q6_V_vzero()), 0);

      {
        sline2c = *iptr++;
        slinewc = *iptr2++;
        sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline2, slinew);
        sline2 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      sout2 = Q6_Vqf32_vmpy_Vqf32Vqf32(sline2, scale_vec);
      sout2 = Q6_Vqf32_vadd_Vqf32Vqf32(sout2, es_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);
      sout2 = Q6_Vsf_vmin_VsfVsf(sout2, high_level_vec);
      sout2 = Q6_Vsf_vmax_VsfVsf(sout2, low_level_vec);
      sout2 = Q6_Vqf32_vmpy_VsfVsf(sout2, round_scale_vec);
      sout2 = Q6_Vsf_equals_Vqf32(sout2);

      sout2 = Q6_Vw_equals_Vsf(sout2);
      sout2 = Q6_Vw_vasr_VwR(sout2, ROUND_2_SCALE);
      // sout2 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout2, Q6_V_vzero()), 0);

      {
        sline3c = *iptr++;
        slinewc = *iptr2++;
        sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline3, slinew);
        sline3 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      sout3 = Q6_Vqf32_vmpy_Vqf32Vqf32(sline3, scale_vec);
      sout3 = Q6_Vqf32_vadd_Vqf32Vqf32(sout3, es_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);
      sout3 = Q6_Vsf_vmin_VsfVsf(sout3, high_level_vec);
      sout3 = Q6_Vsf_vmax_VsfVsf(sout3, low_level_vec);
      sout3 = Q6_Vqf32_vmpy_VsfVsf(sout3, round_scale_vec);
      sout3 = Q6_Vsf_equals_Vqf32(sout3);

      sout3 = Q6_Vw_equals_Vsf(sout3);
      sout3 = Q6_Vw_vasr_VwR(sout3, ROUND_2_SCALE);
      // sout3 = qhmath_hvx_vw_convert_vqf32_rmode(Q6_Vqf32_vadd_VsfVsf(sout3, Q6_V_vzero()), 0);

      {
        sline4c = *iptr++;
        slinewc = *iptr2++;
        sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline4, slinew);
        sline4 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      sout4 = Q6_Vqf32_vmpy_Vqf32Vqf32(sline4, scale_vec);
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

      slinewp = slinewc;
    }
  }

  return 0;
}

int32_t hvx_rmsnorm_auint8_opt(float* restrict input, float* restrict weights, uint8_t* restrict output, uint32_t size,
                               float scale) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)weights;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;
  HVX_Vector sline3p, sline3c, sline3;
  HVX_Vector sline4p, sline4c, sline4;
  HVX_Vector slinewp, slinewc, slinew;

  HVX_Vector zero = Q6_V_vzero();

  int32_t block, l2fetch_block;
  int32_t leftover = size & 31;
  int32_t vectors_in_rounddown = size / 32;
  // int32_t leftover_size = leftover * sizeof(float);

  sline1p = *iptr++;

  // ^2 sum
  HVX_Vector sum = Q6_Vqf32_vadd_VsfVsf(Q6_V_vzero(), Q6_V_vzero());
  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) { l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0); }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

      sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));

      sline1p = sline1c;
    }
  }

  if (vectors_in_rounddown > 0) {
    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
    sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_Vqf32_vmpy_VsfVsf(sline1, sline1));
  }

  float epsilon_ = 1e-6;
  union {
    float f;
    uint32_t ui;
  } sum_value;
  sum_value.f = 0.0f;

  for (int32_t i = 64; i >= 4; i >>= 1) { sum = Q6_Vqf32_vadd_Vqf32Vqf32(sum, Q6_V_vlalign_VVR(sum, zero, i)); }

  sum = Q6_Vsf_equals_Vqf32(sum);
  sum_value.f = 1.0f / sqrtf(*((float*)&sum + 31) / size + epsilon_);

  // x * 1/rsqrt(sum)
  iptr = (HVX_Vector*)input;

  sline1p = *iptr++;
  sline2p = *iptr++;
  sline3p = *iptr++;
  sline4p = *iptr++;

  slinewp = *iptr2++;

  HVX_Vector irsqrt_vsf = Q6_V_vsplat_R(sum_value.ui);
  HVX_Vector irsqrt_vqf32 = Q6_Vqf32_vadd_VsfVsf(irsqrt_vsf, Q6_V_vzero());

  float post_scale_flt = scale / 64.0f;
  int scexp = flt_getexp(post_scale_flt);
  int rsh = min_i32(-scexp, 7);  // e.g. 0.11 -> 0.88, rsh = 3
  float rsh_fac = flt_power2(rsh);

  int adj_bias = roundf_i32(128 * rsh_fac);
  adj_bias = Q6_R_combine_RlRl(adj_bias, adj_bias);

  HVX_Vector zero_v_sf = Q6_V_vzero();
  float es = 0.5f;
  HVX_Vector es_vec = Q6_V_vsplat_R(float_to_fp16s(es));
  es_vec = Q6_Vqf16_vadd_VhfVhf(es_vec, zero_v_sf);

  HVX_Vector vadj = Q6_V_vsplat_R(adj_bias);
  HVX_Vector o_scale_vec = Q6_V_vsplat_R(float_to_fp16s(post_scale_flt * rsh_fac * (1 << 15)));

  for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE) {
    block = Q6_R_min_RR(i, BLOCK_SIZE);
    l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t j = 0; j < block; j += 4) {
      {
        sline1c = *iptr++;
        slinewc = *iptr2++;
        sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline1, slinew);
        sline1 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      {
        sline2c = *iptr++;
        slinewc = *iptr2++;
        sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline2, slinew);
        sline2 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      HVX_Vector sline_low = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(sline2, sline1));
      sline_low = Q6_Vqf16_vadd_Vqf16Vqf16(sline_low, es_vec);
      sline_low = Q6_Vqf16_vmpy_VhfVhf(sline_low, o_scale_vec);
      sline_low = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(sline_low));
      sline_low = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vmpy_VhRh_s1_rnd_sat(sline_low, 0x00400040), vadj);

      sline_low = Q6_Vh_vdeal_Vh(sline_low);

      {
        sline3c = *iptr++;
        slinewc = *iptr2++;
        sline3 = Q6_V_valign_VVR(sline3c, sline3p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline3, slinew);
        sline3 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      {
        sline4c = *iptr++;
        slinewc = *iptr2++;
        sline4 = Q6_V_valign_VVR(sline4c, sline4p, (size_t)input);
        slinew = Q6_V_valign_VVR(slinewc, slinewp, (size_t)weights);

        HVX_Vector middle_value_qf32 = Q6_Vqf32_vmpy_VsfVsf(sline4, slinew);
        sline4 = Q6_Vqf32_vmpy_Vqf32Vqf32(middle_value_qf32, irsqrt_vqf32);

        slinewp = slinewc;
      }

      HVX_Vector sline_high = Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(sline4, sline3));
      sline_high = Q6_Vqf16_vadd_Vqf16Vqf16(sline_high, es_vec);
      sline_high = Q6_Vqf16_vmpy_VhfVhf(sline_high, o_scale_vec);
      sline_high = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(sline_high));
      sline_high = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vmpy_VhRh_s1_rnd_sat(sline_high, 0x00400040), vadj);

      sline_high = Q6_Vh_vdeal_Vh(sline_high);

      HVX_Vector sout = Q6_Vub_vasr_VhVhR_rnd_sat(sline_high, sline_low, rsh);
      sout = Q6_Vb_vdeal_Vb(sout);
      *optr++ = sout;

      sline1p = sline1c;
      sline2p = sline2c;
      sline3p = sline3c;
      sline4p = sline4c;

      slinewp = slinewc;
    }
  }

  return 0;
}

template<typename TensorType>
GraphStatus rmsnormImpl(TensorType& out_0, const TensorType& in_0, const TensorType& weights)

{
  out_0.set_dims(in_0);

  // NHWC

  auto in_ptr = (float*)in_0.raw_data_const();
  auto weights_ptr = (float*)weights.raw_data_const();

  auto [b_in, h_in, w_in, d_in] = in_0.dims();

  DType dtype = out_0.get_dtype();

  if (dtype == DType::Float32) {
    auto out_ptr = (float*)out_0.raw_data();

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          // RMS
          hvx_rmsnorm_af(in_ptr, weights_ptr, out_ptr, d_in);

          in_ptr += d_in;
          out_ptr += d_in;
        }
      }
    }

  } else if (dtype == DType::QUInt8) {
    auto out_ptr = (uint8_t*)out_0.raw_data();
    float scale_ = out_0.interface_scale();

    scale_ = 1.0f / scale_;

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          // RMS
          hvx_rmsnorm_auint8(in_ptr, weights_ptr, out_ptr, d_in, scale_);

          in_ptr += d_in;
          out_ptr += d_in;
        }
      }
    }
  }

  return GraphStatus::Success;
}

#else

template<typename TensorType>
GraphStatus rmsnormImpl(TensorType& out_0, const TensorType& in_0, const TensorType& weights)

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
  // NHWC

  float epsilon_ = 1e-6;
  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // RMS
        float sum_squares = 0.0f;
        for (Idx d = 0; d < d_in; d++) {
          float inval = in_0(b, h, w, d);
          sum_squares += inval * inval;
        }

        // debuglog("silu execute... sum_squares=(%f)", sum_squares);

        float rms = sqrtf(sum_squares / d_in + epsilon_);
        debuglog("rms execute... sum_squares=(%f)", 1.0f / rms);
        debuglog("rms execute... sum_squares=(%f)", sum_squares);

        for (Idx d = 0; d < d_in; d++) {
          float inval = in_0(b, h, w, d);
          float weight = weights(0, 0, 0, d);

          out_0(b, h, w, d) = inval * weight / rms;
        }
      }
    }
  }

  return GraphStatus::Success;
}

#endif

__attribute__((unused)) static float rmsnormCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_RMSNorm);