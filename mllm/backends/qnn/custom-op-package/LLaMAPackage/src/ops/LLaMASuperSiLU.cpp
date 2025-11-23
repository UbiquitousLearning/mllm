//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_LLaMASuperSiLU);

// op execute function declarations
template<typename TensorType>
GraphStatus llamasupersiluImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1,
                               const PlainFloatTensor& a_scale, const PlainFloatTensor& b_scale,
                               const PlainFloatTensor& o_scale);

// forward declaration of sample cost function
static float llamasupersiluCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((llamasupersiluImpl<Tensor>), "LLaMASuperSiLU")
 */
DEF_PACKAGE_OP((llamasupersiluImpl<Tensor>), "LLaMASuperSiLU")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((llamasupersiluImpl<PlainFloatTensor>), "LLaMASuperSiLU", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((llamasupersiluImpl<PlainFloatTensor>),
 * "LLaMASuperSiLU", llamasupersiluCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("LLaMASuperSiLU", "a_scale", true, nullptr, "b_scale", true, nullptr, "o_scale", true, nullptr)

/* execute functions for ops */

#ifndef REFERENCE_OP

#include <hexagon_types.h>
#include "hvx_internal.h"
#include <stddef.h>

#define BLOCK_SIZE (8 * 1024 / VLEN) /* vector chunks */
#define L2FETCH_AHEAD (BLOCK_SIZE)

#define FP16_MANTISA 10
#define FP16_EXPONENT_MASK 0x1f
#define FP16_EXPONENT_BIAS 0xf
#define FP16_MANTISA_MASK 0x000003ff
#define FP16_SIGN 15
#define FP16_NEG_1 0xbc00
#define ROUND_2_SCALE 22
#define ROUND_SCALSE ((1 << ROUND_2_SCALE) * 1.0f)

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

static const float fp16_c0_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.13239719960243818,
    0.2216255210749415,
    0.3447664743728659,
    0.48137452032585476,
    0.5716299228719798,
    0.5547323231605259,
    0.5046287748870234,
    0.4999985574626892,
    0.5000036514755082,
    0.49475652448004626,
    0.4441393352532763,
    0.428500379952032,
    0.5173297285470642,
    0.6541461039833616,
    0.7783931007462818,
    0.8678015179911097,
};
static const float fp16_c1_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.05928005756790343,
    0.11063222460270064,
    0.1932879057003057,
    0.30302440212086995,
    0.3922924462181049,
    0.36546332659415875,
    0.2644148210990377,
    0.24989020912329707,
    0.2498532691910313,
    0.2661055781198988,
    0.36728015359480604,
    0.39215270010450015,
    0.3041825601732039,
    0.1940762094668647,
    0.11061794856987572,
    0.059174800917353595,
};
static const float fp16_c2_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.010145494303219278,
    0.02123968384425681,
    0.04207468332514667,
    0.07519946712591977,
    0.10840620196267145,
    0.09270738184406795,
    0.015322371881818012,
    -0.0009948273994921822,
    0.0011544907060402412,
    -0.017040517565094934,
    -0.09379878876657094,
    -0.10835043868732394,
    -0.07558705272699548,
    -0.04228875316413285,
    -0.021235740718738055,
    -0.010124599879590107,
};
static const float fp16_c3_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0007841223015974933,
    0.001850453397354219,
    0.004187899308371771,
    0.008640952434084206,
    0.01414741414964877,
    0.010117749275618,
    -0.01654848996354919,
    -0.02395108399453624,
    -0.024199111971064446,
    -0.015783556879607072,
    0.010407672131558174,
    0.014137608186323335,
    0.008698510795258909,
    0.004213708431213342,
    0.0018499827774393985,
    0.0007822799742289481,
};
static const float fp16_c4_coeffs[32] __attribute__((aligned(VLEN))) = {
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.3031641204975905e-05,
    6.150442488966733e-05,
    0.00015997783736818624,
    0.00038491646239693526,
    0.0007283649599237781,
    0.00034439150914392054,
    -0.003142246198646662,
    -0.004120389580321761,
    0.004246050162553198,
    0.0030162727520777893,
    -0.00037312974308425725,
    -0.0007277242855014247,
    -0.00038811687679772674,
    -0.0001611434776868886,
    -6.14837984586862e-05,
    -2.297076123375133e-05,
};

int32_t hvx_supersilu_ahf(uint8_t* restrict input, uint8_t* restrict input2, uint8_t* restrict output, float a_scale,
                          float b_scale, float o_scale, uint32_t size) {
  if ((input == NULL) || (output == NULL) || (size == 0)) { return -1; }

  HVX_Vector* iptr = (HVX_Vector*)input;
  HVX_Vector* iptr2 = (HVX_Vector*)input2;
  HVX_UVector* optr = (HVX_UVector*)output;
  HVX_Vector sline1p, sline1c, sline1;
  HVX_Vector sline2p, sline2c, sline2;

  int32_t block, l2fetch_block;
  int32_t leftover = size & 128;
  int32_t vectors_in_rounddown = size / 128;
  // int32_t leftover_size = leftover * sizeof(__fp16);

  sline1p = *iptr++;
  sline2p = *iptr2++;

  // dequantize
  uint32_t convert = 0x00800080;
  HVX_Vector convert_vector = Q6_V_vsplat_R(convert);

  HVX_Vector a_scale_vec = Q6_V_vsplat_R(float_to_fp16s(a_scale));
  HVX_Vector b_scale_vec = Q6_V_vsplat_R(float_to_fp16s(b_scale));
  HVX_Vector zero_v_sf = Q6_V_vzero();

  // silu
  HVX_Vector input_min_v_hf;
  HVX_Vector input_shifted_v_hf;
  HVX_Vector input_scaled_v;
  HVX_VectorPair input_vp_qf32;
  // HVX_Vector input_v_qf16;
  HVX_Vector mask_idx1_v, mask_idx2_v;
  HVX_Vector const16_0_v_hf;
  HVX_Vector zero_v_hf, one_v_hf;
  HVX_Vector tmp_v;
  HVX_Vector idx1_v, idx2_v;
  HVX_Vector scale_v;
  HVX_DV output_dv;
  HVX_DV c0_coeff_dv;
  HVX_VectorPair c0_coeff_vp;
  HVX_Vector c0_coeff_v;
  HVX_DV c1_coeff_dv;
  HVX_VectorPair c1_coeff_vp;
  HVX_Vector c1_coeff_v;
  HVX_DV c2_coeff_dv;
  HVX_VectorPair c2_coeff_vp;
  HVX_Vector c2_coeff_v;
  HVX_DV c3_coeff_dv;
  HVX_VectorPair c3_coeff_vp;
  HVX_Vector c3_coeff_v;
  HVX_DV c4_coeff_dv;
  HVX_VectorPair c4_coeff_vp;
  HVX_Vector c4_coeff_v;

  scale_v = Q6_Vh_vsplat_R(0x3bfe);

  /* Vector of ones used as mpy neutral element in conversions from hf vector to qf32 vector pair */
  one_v_hf = Q6_Vh_vsplat_R(0x3c00);

  /*
   * Vector of zeroes used as neutral element in hf to qf16 conversions.
   * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
   *       can be avoided in real-time, but this is not done in order to don't
   *       sacrify code readibility in expense of insignificant performance improvement.
   */
  zero_v_hf = Q6_V_vzero();

  /* Mask for extracting only 4 bits of mantissa */
  mask_idx1_v = Q6_Vh_vsplat_R(0x000F);

  mask_idx2_v = Q6_V_vsplat_R(0x00001010);

  /* 16.0 in IEEE 16-bit floating-point representation */
  const16_0_v_hf = Q6_Vh_vsplat_R(0x4c00);

  /*
   * Prepare vector of input_min values, that is used later in shifting input range.
   * input_min is low boundary of specified input range.
   */
  input_min_v_hf = Q6_Vh_vsplat_R(0xc800);

  /* Convert scale factor from hf to q16. Use the same vector for both formats */
  scale_v = Q6_Vqf16_vadd_VhfVhf(scale_v, zero_v_hf);

  /* Load coefficients */
  c0_coeff_v = *((HVX_Vector*)(fp16_c0_coeffs));
  c1_coeff_v = *((HVX_Vector*)(fp16_c1_coeffs));
  c2_coeff_v = *((HVX_Vector*)(fp16_c2_coeffs));
  c3_coeff_v = *((HVX_Vector*)(fp16_c3_coeffs));
  c4_coeff_v = *((HVX_Vector*)(fp16_c4_coeffs));

  /* Convert coefficients from hf to qf32 format. Use the same vector for both representations */
  c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_hf);
  c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_hf);
  c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_hf);
  c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_hf);
  c4_coeff_v = Q6_Vqf32_vadd_VsfVsf(c4_coeff_v, zero_v_hf);

  /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
  c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
  c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
  c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
  c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);
  c4_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c4_coeff_v);

  // quantize
  HVX_Vector low_level_vec, high_level_vec, o_scale_vec, es_vec, round_scale_vec;
  HVX_Vector uintconvert = Q6_V_vsplat_R(0x80808080);
  HVX_Vector vmb = Q6_V_vsplat_R(0x40004000);

  float post_scale_flt = a_scale * b_scale * o_scale;
  int scexp = flt_getexp(post_scale_flt);
  int rsh = min_i32(-scexp, 7);  // e.g. 0.11 -> 0.88, rsh = 3
  float rsh_fac = flt_power2(rsh);

  int adj_bias = roundf_i32(128 * rsh_fac);
  adj_bias = Q6_R_combine_RlRl(adj_bias, adj_bias);

  HVX_Vector vadj = Q6_V_vsplat_R(adj_bias);

  float es = 0.5;
  low_level_vec = Q6_V_vsplat_R(float_to_fp16s(-128.0f));
  high_level_vec = Q6_V_vsplat_R(float_to_fp16s(127.0f));
  o_scale_vec = Q6_V_vsplat_R(float_to_fp16s(post_scale_flt * rsh_fac * (1 << 15)));
  // one_vec = Q6_V_vsplat_R(float_to_fp16s(1.0f));
  // o_scale_vec = Q6_Vqf16_vadd_VhfVhf(o_scale_vec, zero_v_hf);
  es_vec = Q6_V_vsplat_R(float_to_fp16s(es));
  round_scale_vec = Q6_V_vsplat_R(float_to_bits(ROUND_SCALSE));

  es_vec = Q6_Vqf16_vadd_VhfVhf(es_vec, zero_v_sf);
  round_scale_vec = Q6_Vqf32_vadd_VsfVsf(round_scale_vec, zero_v_sf);

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

    if (l2fetch_block > 0) {
      l2fetch(iptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
      l2fetch(iptr2 + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
    }

    for (int32_t j = 0; j < block; ++j) {
      sline1c = *iptr++;
      sline2c = *iptr2++;
      sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);
      sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input2);

      HVX_Vector sline1_high;
      HVX_Vector sline1_low;
      // HVX_Vector sline2_high;
      // HVX_Vector sline2_low;

      {
        // dequantize  sline1 qf16
        HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

        temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
        HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
        HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

        sline1_low = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), a_scale_vec);
        sline1_low = Q6_Vhf_equals_Vqf16(sline1_low);
        sline1_high = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), a_scale_vec);
        sline1_high = Q6_Vhf_equals_Vqf16(sline1_high);
      }

      {
        // silu  sline1_low
        tmp_v = Q6_Vh_vdeal_Vh(sline1_low);

        /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
        input_shifted_v_hf = Q6_Vqf16_vsub_VhfVhf(tmp_v, input_min_v_hf);

        /*
         * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
         * in order to get corresponding coefficient indexes
         */
        input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);

        /*
         * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
         * to [16.0,32.0) in order to convert float indexes to integer values.
         * Float values, represented in IEEE 754, in range [16.0,32.0] have the
         * same exponent, which means 4 MSB of mantissa carry information about
         * integer index.
         * Use the same input_scaled_v vector for hf and qf16 representation
         */
        input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);

        /* Convert back from qf16 to hf in order to extract integer index  */
        tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

        /* Only 4 MSB bits of mantissa represent segment index */
        idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);

        /* Ensure only 4 MSB bits of mantissa are used as indexes */
        idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);

        idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
        idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
        idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

        /* Obtain the polynomial coefficients from lookup table */
        c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
        c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);

        /* Convert input from hf vector to qf32 vector pair for Horner's method*/
        input_vp_qf32 = Q6_Wqf32_vmpy_VhfVhf(sline1_low, one_v_hf);

        /* Perform evaluation of polynomial using Horner's method */
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c3_coeff_vp));
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c2_coeff_vp));
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c1_coeff_vp));
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c0_coeff_vp));

        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(c4_coeff_vp), Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c3_coeff_vp));
        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c2_coeff_vp));
        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c1_coeff_vp));
        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c0_coeff_vp));

        // x * sigmod
        // output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(input_vp_qf32), output_dv.V.lo);
        // output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(input_vp_qf32), output_dv.V.hi);

        sline1_low = Q6_Vhf_equals_Wqf32(output_dv.VV);
      }

      {
        // silu  sline1_high
        tmp_v = Q6_Vh_vdeal_Vh(sline1_high);

        /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
        input_shifted_v_hf = Q6_Vqf16_vsub_VhfVhf(tmp_v, input_min_v_hf);

        /*
         * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
         * in order to get corresponding coefficient indexes
         */
        input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);

        /*
         * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
         * to [16.0,32.0) in order to convert float indexes to integer values.
         * Float values, represented in IEEE 754, in range [16.0,32.0] have the
         * same exponent, which means 4 MSB of mantissa carry information about
         * integer index.
         * Use the same input_scaled_v vector for hf and qf16 representation
         */
        input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);

        /* Convert back from qf16 to hf in order to extract integer index  */
        tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

        /* Only 4 MSB bits of mantissa represent segment index */
        idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);

        /* Ensure only 4 MSB bits of mantissa are used as indexes */
        idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);

        idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
        idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
        idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

        /* Obtain the polynomial coefficients from lookup table */
        c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
        c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
        c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
        c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
        c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
        c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);

        /* Convert input from hf vector to qf32 vector pair for Horner's method*/
        input_vp_qf32 = Q6_Wqf32_vmpy_VhfVhf(sline1_high, one_v_hf);

        /* Perform evaluation of polynomial using Horner's method */
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c3_coeff_vp));
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c2_coeff_vp));
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c1_coeff_vp));
        output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
        output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c0_coeff_vp));

        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(c4_coeff_vp), Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c3_coeff_vp));
        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c2_coeff_vp));
        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c1_coeff_vp));
        output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
        output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c0_coeff_vp));

        // x * sigmod
        // output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(input_vp_qf32), output_dv.V.lo);
        // output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(input_vp_qf32), output_dv.V.hi);

        sline1_high = Q6_Vhf_equals_Wqf32(output_dv.VV);
      }

      HVX_Vector sline_high;
      HVX_Vector sline_low;

      HVX_VectorPair mul_output;
      {
        // uint8 mul
        // (a-128)*(b-128) = a*b - 128 (a+b) + 128*128
        HVX_VectorPair prod1 = Q6_Wuh_vmpyacc_WuhVubVub(Q6_W_vcombine_VV(vmb, vmb), sline1, sline2);
        HVX_VectorPair prod2 = Q6_Wh_vmpa_WubRub(Q6_W_vcombine_VV(sline2, sline1), 0x80808080);
        mul_output = Q6_Wh_vsub_WhWh(prod1, prod2);

        mul_output = Q6_W_vshuff_VVR(Q6_V_hi_W(mul_output), Q6_V_lo_W(mul_output), -2);
      }

      {
        // scaling quantize
        sline_low = Q6_Vqf16_vmpy_VhfVhf(sline1_low, o_scale_vec);
        sline_low = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(sline_low));
        sline_low = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vmpy_VhVh_s1_rnd_sat(Q6_V_lo_W(mul_output), sline_low), vadj);

        sline_high = Q6_Vqf16_vmpy_VhfVhf(sline1_high, o_scale_vec);
        sline_high = Q6_Vh_equals_Vhf(Q6_Vhf_equals_Vqf16(sline_high));
        sline_high = Q6_Vh_vadd_VhVh_sat(Q6_Vh_vmpy_VhVh_s1_rnd_sat(sline_high, Q6_V_hi_W(mul_output)), vadj);

        HVX_Vector sout = Q6_Vub_vasr_VhVhR_rnd_sat(sline_high, sline_low, rsh);
        sout = Q6_Vb_vdeal_Vb(sout);
        *optr++ = sout;
      }

      sline1p = sline1c;
      sline2p = sline2c;
    }
  }

  if (vectors_in_rounddown > 0) {
    o_scale_vec = Q6_V_vsplat_R(float_to_fp16s(o_scale));

    sline1c = is_aligned(iptr, VLEN) && leftover == 0 ? sline1p : *iptr++;
    sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

    sline2c = is_aligned(iptr2, VLEN) && leftover == 0 ? sline2p : *iptr2++;
    sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input2);

    HVX_Vector sline1_high;
    HVX_Vector sline1_low;
    HVX_Vector sline2_high;
    HVX_Vector sline2_low;

    {
      // dequantize  sline1 qf16
      HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline1, zero_v_sf);

      temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
      HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

      sline1_low = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), a_scale_vec);
      sline1_low = Q6_Vhf_equals_Vqf16(sline1_low);
      sline1_high = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), a_scale_vec);
      sline1_high = Q6_Vhf_equals_Vqf16(sline1_high);
    }

    {
      // dequantize  sline2 qf16
      HVX_VectorPair temp = Q6_Wh_vadd_VubVub(sline2, zero_v_sf);

      temp = Q6_W_vshuff_VVR(Q6_V_hi_W(temp), Q6_V_lo_W(temp), -2);
      HVX_Vector sout1 = Q6_Vh_vsub_VhVh(Q6_V_lo_W(temp), convert_vector);
      HVX_Vector sout2 = Q6_Vh_vsub_VhVh(Q6_V_hi_W(temp), convert_vector);

      sline2_low = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout1), b_scale_vec);
      sline2_low = Q6_Vhf_equals_Vqf16(sline2_low);
      sline2_high = Q6_Vqf16_vmpy_VhfVhf(Q6_Vhf_equals_Vh(sout2), b_scale_vec);
      sline2_high = Q6_Vhf_equals_Vqf16(sline2_high);
    }

    {
      // silu  sline1_low
      tmp_v = Q6_Vh_vdeal_Vh(sline1_low);

      /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
      input_shifted_v_hf = Q6_Vqf16_vsub_VhfVhf(tmp_v, input_min_v_hf);

      /*
       * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
       * in order to get corresponding coefficient indexes
       */
      input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);

      /*
       * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
       * to [16.0,32.0) in order to convert float indexes to integer values.
       * Float values, represented in IEEE 754, in range [16.0,32.0] have the
       * same exponent, which means 4 MSB of mantissa carry information about
       * integer index.
       * Use the same input_scaled_v vector for hf and qf16 representation
       */
      input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);

      /* Convert back from qf16 to hf in order to extract integer index  */
      tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

      /* Only 4 MSB bits of mantissa represent segment index */
      idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);

      /* Ensure only 4 MSB bits of mantissa are used as indexes */
      idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);

      idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
      idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
      idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

      /* Obtain the polynomial coefficients from lookup table */
      c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
      c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
      c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
      c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
      c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
      c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
      c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
      c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
      c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
      c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);

      /* Convert input from hf vector to qf32 vector pair for Horner's method*/
      input_vp_qf32 = Q6_Wqf32_vmpy_VhfVhf(sline1_low, one_v_hf);

      /* Perform evaluation of polynomial using Horner's method */
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c3_coeff_vp));
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c2_coeff_vp));
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c1_coeff_vp));
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c0_coeff_vp));

      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(c4_coeff_vp), Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c3_coeff_vp));
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c2_coeff_vp));
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c1_coeff_vp));
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c0_coeff_vp));

      // x * sigmod
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(input_vp_qf32), output_dv.V.lo);
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(input_vp_qf32), output_dv.V.hi);

      sline1_low = Q6_Vhf_equals_Wqf32(output_dv.VV);
    }

    {
      // silu  sline1_high
      tmp_v = Q6_Vh_vdeal_Vh(sline1_high);

      /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
      input_shifted_v_hf = Q6_Vqf16_vsub_VhfVhf(tmp_v, input_min_v_hf);

      /*
       * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
       * in order to get corresponding coefficient indexes
       */
      input_scaled_v = Q6_Vqf16_vmpy_Vqf16Vqf16(input_shifted_v_hf, scale_v);

      /*
       * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
       * to [16.0,32.0) in order to convert float indexes to integer values.
       * Float values, represented in IEEE 754, in range [16.0,32.0] have the
       * same exponent, which means 4 MSB of mantissa carry information about
       * integer index.
       * Use the same input_scaled_v vector for hf and qf16 representation
       */
      input_scaled_v = Q6_Vqf16_vadd_Vqf16Vhf(input_scaled_v, const16_0_v_hf);

      /* Convert back from qf16 to hf in order to extract integer index  */
      tmp_v = Q6_Vhf_equals_Vqf16(input_scaled_v);

      /* Only 4 MSB bits of mantissa represent segment index */
      idx1_v = Q6_Vuh_vlsr_VuhR(tmp_v, 6);

      /* Ensure only 4 MSB bits of mantissa are used as indexes */
      idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);

      idx1_v = Q6_Vb_vshuff_Vb(idx1_v);
      idx1_v = Q6_V_vor_VV(idx1_v, mask_idx2_v);
      idx2_v = Q6_Vw_vasl_VwR(idx1_v, 16);

      /* Obtain the polynomial coefficients from lookup table */
      c0_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c0_coeff_dv.VV), 1);
      c0_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c0_coeff_vp, idx2_v, Q6_V_hi_W(c0_coeff_dv.VV), 1);
      c1_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c1_coeff_dv.VV), 1);
      c1_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c1_coeff_vp, idx2_v, Q6_V_hi_W(c1_coeff_dv.VV), 1);
      c2_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c2_coeff_dv.VV), 1);
      c2_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c2_coeff_vp, idx2_v, Q6_V_hi_W(c2_coeff_dv.VV), 1);
      c3_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c3_coeff_dv.VV), 1);
      c3_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c3_coeff_vp, idx2_v, Q6_V_hi_W(c3_coeff_dv.VV), 1);
      c4_coeff_vp = Q6_Wh_vlut16_VbVhR(idx1_v, Q6_V_lo_W(c4_coeff_dv.VV), 1);
      c4_coeff_vp = Q6_Wh_vlut16or_WhVbVhR(c4_coeff_vp, idx2_v, Q6_V_hi_W(c4_coeff_dv.VV), 1);

      /* Convert input from hf vector to qf32 vector pair for Horner's method*/
      input_vp_qf32 = Q6_Wqf32_vmpy_VhfVhf(sline1_high, one_v_hf);

      /* Perform evaluation of polynomial using Horner's method */
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c3_coeff_vp));
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c2_coeff_vp));
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c1_coeff_vp));
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(input_vp_qf32));
      output_dv.V.lo = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.lo, Q6_V_lo_W(c0_coeff_vp));

      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(c4_coeff_vp), Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c3_coeff_vp));
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c2_coeff_vp));
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c1_coeff_vp));
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(input_vp_qf32));
      output_dv.V.hi = Q6_Vqf32_vadd_Vqf32Vqf32(output_dv.V.hi, Q6_V_hi_W(c0_coeff_vp));

      // x * sigmod
      output_dv.V.lo = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(input_vp_qf32), output_dv.V.lo);
      output_dv.V.hi = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_hi_W(input_vp_qf32), output_dv.V.hi);

      sline1_high = Q6_Vhf_equals_Wqf32(output_dv.VV);
    }

    HVX_Vector sline_high;
    HVX_Vector sline_low;

    {
      // mul
      sline_high = Q6_Vqf16_vmpy_VhfVhf(sline1_high, sline2_high);
      sline_low = Q6_Vqf16_vmpy_VhfVhf(sline1_low, sline2_low);

      sline_high = Q6_Vhf_equals_Vqf16(sline_high);
      sline_low = Q6_Vhf_equals_Vqf16(sline_low);
    }

    {
      // quantize
      HVX_Vector sout1 = Q6_Vqf16_vmpy_VhfVhf(sline_low, o_scale_vec);
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

      HVX_Vector sout2 = Q6_Vqf16_vmpy_VhfVhf(sline_high, o_scale_vec);
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

      HVX_Vector reql_h = Q6_Vb_vpack_VhVh_sat(sout2, sout1);
      *optr++ = Q6_Vb_vadd_VbVb(reql_h, uintconvert);
    }
  }

  // // Handle leftover elements.
  // if (leftover_size > 0) {
  //   sline1c = (is_in_one_chunk(iptr, leftover_size, VLEN)
  //                   ? sline1p
  //                   : *iptr++);
  //   sline1 = Q6_V_valign_VVR(sline1c, sline1p, (size_t)input);

  //   sline2c = (is_in_one_chunk(iptr2, leftover_size, VLEN)
  //                   ? sline2p
  //                   : *iptr2++);
  //   sline2 = Q6_V_valign_VVR(sline2c, sline2p, (size_t)input2);

  //   vstu_variable(optr, leftover_size,  Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(sline1, sline2)));
  // }

  return 0;
}

template<typename TensorType>
GraphStatus llamasupersiluImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1,
                               const PlainFloatTensor& a_scale, const PlainFloatTensor& b_scale,
                               const PlainFloatTensor& o_scale)

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
  size_t size = b_in * h_in * w_in * d_in;

  float a_scale_ = a_scale(0, 0, 0, 0);
  float b_scale_ = b_scale(0, 0, 0, 0);
  float o_scale_ = o_scale(0, 0, 0, 0);

  auto in_ptr = (uint8_t*)in_0.raw_data_const();
  auto in_ptr2 = (uint8_t*)in_1.raw_data_const();

  auto out_ptr = (uint8_t*)out_0.raw_data();

  DType dtype = in_0.get_dtype();

  if (dtype == DType::QUInt8 && out_0.get_dtype() == DType::QUInt8) {
    hvx_supersilu_ahf(in_ptr, in_ptr2, out_ptr, a_scale_, b_scale_, 1.0f / o_scale_, size);
  }

  return GraphStatus::Success;
}

#else

template<typename TensorType>
GraphStatus llamasupersiluImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1,
                               const PlainFloatTensor& a_scale, const PlainFloatTensor& b_scale,
                               const PlainFloatTensor& o_scale)

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

  float a_scale_ = a_scale(0, 0, 0, 0);
  float b_scale_ = b_scale(0, 0, 0, 0);
  float o_scale_ = o_scale(0, 0, 0, 0);

  auto in_ptr = (uint8_t*)in_0.raw_data_const();
  auto in_ptr2 = (uint8_t*)in_1.raw_data_const();

  auto out_ptr = (uint8_t*)out_0.raw_data();

  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // mul
        for (Idx d = 0; d < d_in; d++) {
          int32_t a_inval = static_cast<int32_t>(*in_ptr++);
          float a_inval_fp16 = (a_inval - 128) * a_scale_;

          int32_t b_inval = static_cast<int32_t>(*in_ptr2++);
          float b_inval_fp16 = (b_inval - 128) * b_scale_;

          a_inval_fp16 = a_inval_fp16 * (1 / (1 + expf(-a_inval_fp16)));

          float inval = a_inval_fp16 * b_inval_fp16;

          long v = lroundf(inval / o_scale_);

          if (v > 127) v = 127;

          if (v < -128) v = -128;

          v += 128;

          *out_ptr++ = static_cast<uint8_t>(v);
        }
      }
    }
  }

  return GraphStatus::Success;
}

#endif

__attribute__((unused)) static float llamasupersiluCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_LLaMASuperSiLU);