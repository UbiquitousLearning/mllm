//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_SiLU);


// op execute function declarations
template<typename TensorType>
GraphStatus siluImpl(TensorType& out_0,
                     const TensorType& in_0);

// forward declaration of sample cost function
static float siluCostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((siluImpl<Tensor>), "SiLU")
 */
DEF_PACKAGE_OP((siluImpl<Tensor>), "SiLU")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((siluImpl<PlainFloatTensor>), "SiLU", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((siluImpl<PlainFloatTensor>),
 * "SiLU", siluCostFunc, Flags::RESOURCE_HVX)
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
/**=============================================================================
@file
    qhmath_hvx_sigmoid_af.c

@brief
    Calculate polynomial approximation of the function below in
    floating-point arithmetic using HVX instructions.

    Function: 1.0/(exp(-x)+1.0)

    Function is approximated in specified input range from -8.0 to 8.0,
    where inputs and outputs are arrays of 32-bit float values.

    Approximation is performed using the following method:

    1) Input range is split into 16 equidistant segments
    2) For each segment, Numpy’s polynomial package is used to find the best
       polynomial approximation of order N with the corresponding C0, C1, ..., Cn.
    3) VLUT instructions are used to select appropriate coefficients for each input sample
    4) Horner's method is used to compute polynomial values:
       f(x) = ((((Cn*x + Cn-1)*x + Cn-2)*x + ...)*x + C1)*x + C0

Copyright (c) 2020 Qualcomm Technologies Incorporated.
All Rights Reserved. Qualcomm Proprietary and Confidential.
=============================================================================**/

#include "qhmath_hvx.h"
#include "hvx_internal.h"
#include <hexagon_types.h>
#include <stddef.h>

#define BLOCK_SIZE       (8*1024/VLEN)  /* vector chunks */
#define L2FETCH_AHEAD    (BLOCK_SIZE)

/* Polynomial coefficients */
static const float c0_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
0.1329913082916337,0.22308514882873062,0.347752862580421,0.4845759228057826,0.5724725619240282,0.5532613332075828,0.5041402176920755,0.4999998945071365,
0.500005251569411,0.494975832882496,0.44426898861108216,0.42865769845972046,0.5186084804556764,0.6556781472810073,0.7780379623543565,0.8670752648575938,
};
static const float c1_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
0.0595948414501292,0.11153317908159224,0.19545701719511055,0.3058925677063833,0.3932668307015573,0.3630691859433203,0.26302954631996744,0.2499155333713503,
0.24983690256810576,0.26551386754654915,0.3670764533308477,0.39196882072648825,0.3030372911476408,0.19296191313371913,0.11084562978488391,0.059559556604464964,
};
static const float c2_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
0.010207999856103376,0.02144807112969563,0.04266485934992188,0.07616157468726052,0.10882760873715347,0.09125379784995667,0.013872106909816257,-0.0008786208359828815,
0.0011993845621092196,-0.01645080326288375,-0.09367947263571219,-0.10827006684348266,-0.07520301291634655,-0.04198514892887826,-0.021290356584896874,-0.010200991240527542,
};
static const float c3_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
0.0007896351019423816,0.0018718593077865326,0.004259190313167949,0.008784166436796144,0.014228201960903939,0.009727536748893095,-0.01721317464724529,-0.023762851116001377,
-0.02424226654277249,-0.01604104065157868,0.010376786273973133,0.014122038833203628,0.008641365746408176,0.004176981844803722,0.0018557930308154783,0.0007890167735032168,
};
static const float c4_coeffs[32] __attribute__((aligned(VLEN))) =
{
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,       0.0,
2.3213858349988003e-05,6.232838199801025e-05,0.0001632037964535633,0.0003928983460811959,0.0007341577078787206,0.0003053082875419616,-0.003254838747910248,-0.004021655986643196,
0.004258314078650583,0.0030578644020607566,-0.00037014803880675387,-0.0007265964578827031,-0.0003849331969038772,-0.00015947916435728337,-6.171511304866758e-05,-2.319341439172678e-05,
};

/**
 * @brief       Polynomial approximation of x/(exp(-x)+1.0) function.
 * @param[in]   input   Input array of elements in IEEE 32-bit floating-point format.
 * @param[out]  output  Output array of elements in IEEE 32-bit floating-point format.
 * @param[in]   length  Number of elements in input/output arrays.
 * @return      Returns 0 on successful execution. Otherwise -1.
 */
int32_t hvx_silu_af(float *restrict input, float *restrict output, uint32_t size)
{
    HVX_Vector *input_v_ptr;
    HVX_UVector *output_v_ptr;
    HVX_Vector input_min_v_f;
    HVX_Vector input_shifted_v_qf32;
    HVX_Vector input_scaled_v_qf32;
    HVX_Vector scale_v;
    HVX_Vector input_v_qf32;
    HVX_Vector const16_0_v_sf;
    HVX_Vector zero_v_sf;
    HVX_Vector mask_idx1_v, mask_idx2_v;
    HVX_Vector tmp_v, idx1_v, idx2_v;
    HVX_Vector output_v;
    HVX_Vector slinep;
    HVX_Vector slinec;
    HVX_Vector sline;
    HVX_Vector sout;
    int32_t block, l2fetch_block;
    int32_t leftover = size & 31;
    int32_t vectors_in_rounddown = size / 32;
    int32_t leftover_size = leftover * sizeof(float);
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

    /* Check input arguments. Return error status if some argument has invalid value */
    if ((input == 0) || (output == 0) || (size == 0))
    {
        return -1;
    }

    input_v_ptr = (HVX_Vector *) input;
    output_v_ptr = (HVX_UVector *) output;

    /*
     * If input data is not aligned to HVX vector size, compose aligned vectors
     * from data loaded in slinep and slinec
     */
    slinep = *input_v_ptr++;

    /*
     * Splat scale factor in order to be used later for finding indexes of coefficients.
     * Scale factor is represented in IEEE 16-bit floating-point format and it is
     * calculated using the following formula:
     *    scale_factor = (16.0 / (b0 - a0))
     * NOTE: Calculated value is slightly decreased in order to avoid out of bound
     *       indexes during VLUT lookup.
     */
    scale_v = Q6_V_vsplat_R(0x3f7ffffe);

    /*
     * Vector of zeroes used as neutral element in sf to qf32 conversions.
     * NOTE: Some of conversions (i.e conversion of scale factor and coefficients)
     *       can be avoided in real-time, but this is not done in order to don't
     *       sacrify code readibility in expense of insignificant performance improvement.
     */
    zero_v_sf = Q6_V_vzero();

    /* Mask for extracting only 4 bits of mantissa */
    mask_idx1_v = Q6_V_vsplat_R(0x0000000F);
    mask_idx2_v = Q6_V_vsplat_R(0x00000010);

    /* 16.0 in IEEE 16-bit floating-point representation */
    const16_0_v_sf = Q6_V_vsplat_R(0x41800000);

    /*
     * Prepare vector of input_min values, that is used later in shifting input range.
     * input_min is low boundary of specified input range.
     */
    input_min_v_f = Q6_V_vsplat_R(0xc1000000);

    /* Convert scale factor from sf to q32. Use the same vector for both formats */
    scale_v = Q6_Vqf32_vadd_VsfVsf(scale_v, zero_v_sf);

    /* Load coefficients */
    c0_coeff_v = *((HVX_Vector *)(c0_coeffs));
    c1_coeff_v = *((HVX_Vector *)(c1_coeffs));
    c2_coeff_v = *((HVX_Vector *)(c2_coeffs));
    c3_coeff_v = *((HVX_Vector *)(c3_coeffs));
    c4_coeff_v = *((HVX_Vector *)(c4_coeffs));

    /* Convert coefficients from sf to qf32 format. Use the same vector for both representations */
    c0_coeff_v = Q6_Vqf32_vadd_VsfVsf(c0_coeff_v, zero_v_sf);
    c1_coeff_v = Q6_Vqf32_vadd_VsfVsf(c1_coeff_v, zero_v_sf);
    c2_coeff_v = Q6_Vqf32_vadd_VsfVsf(c2_coeff_v, zero_v_sf);
    c3_coeff_v = Q6_Vqf32_vadd_VsfVsf(c3_coeff_v, zero_v_sf);
    c4_coeff_v = Q6_Vqf32_vadd_VsfVsf(c4_coeff_v, zero_v_sf);

    /* Split 32-bit coefficients to lower and upper part in order to obtain them later with VLUT16. */
    c0_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c0_coeff_v);
    c1_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c1_coeff_v);
    c2_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c2_coeff_v);
    c3_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c3_coeff_v);
    c4_coeff_dv.VV = Q6_Wuw_vzxt_Vuh(c4_coeff_v);

    /*
     * Handle number of whole vectors in input data.
     * Don't process last vector in order to avoid out-of-boundary load.
     */
    for (int32_t i = vectors_in_rounddown - 1; i > 0; i -= BLOCK_SIZE)
    {
        block = Q6_R_min_RR(i, BLOCK_SIZE);
        l2fetch_block = Q6_R_min_RR(i - L2FETCH_AHEAD, BLOCK_SIZE);

        if (l2fetch_block > 0)
        {
            l2fetch(input_v_ptr + L2FETCH_AHEAD, VLEN, VLEN, l2fetch_block, 0);
        }

        /* Process one vector at a time */
        for (int32_t j = 0; j < block; ++j)
        {
            slinec = *input_v_ptr++;

            /* Compose vector of input data from slinec and slinep */
            sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

            /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
            input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline, input_min_v_f);

            /*
             * Scale shifted input range from [0, input_max - input_min] to [0,16.0)
             * in order to get corresponding coefficient indexes
             */
            input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

            /*
             * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
             * to [16.0,32.0) in order to convert float indexes to integer values.
             * Float values, represented in IEEE 754, in range [16.0,32.0] have the
             * same exponent, which means 4 MSB of mantissa carry information about
             * integer index.
             */
            input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

            /* Convert back from qf32 to sf in order to extract integer index */
            tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

            /* Only 4 MSB bits of mantissa represent segment index */
            idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

            idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
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

            /* Convert input from sf vector to qf32 vector for Horner's method*/
            input_v_qf32 = Q6_Vqf32_vadd_VsfVsf(sline, zero_v_sf);

            /* Perform evaluation of polynomial using Horner's method */
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c3_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
            output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

            // x * sigmod
            output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(input_v_qf32, output_v);
            

            /* Store results to the output buffer and convert from qf32 to sf */
            *((HVX_UVector *)(output_v_ptr++)) = Q6_Vsf_equals_Vqf32(output_v);

            /* Prepare slinep for next iteration */
            slinep = slinec;
        }
    }

    /* Handle last whole vector from input data */
    if (vectors_in_rounddown > 0)
    {
        slinec = is_aligned(input_v_ptr, VLEN) && leftover == 0 ? slinep : *input_v_ptr++;
        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

        /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
        input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline, input_min_v_f);

        /* Scale shifted input range from [0, input_max - input_min] to [0,16.0) */
        input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

        /*
         * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
         * to [16.0,32.0) in order to convert float indexes to integer values.
         * Float values, represented in IEEE 754, in range [16.0,32.0] have the
         * same exponent, which means 4 MSB of mantissa carry information about
         * integer index.
         */
        input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

        /* Convert back from qf32 to sf in order to extract integer index */
        tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

        /* Only 4 MSB bits of mantissa represent segment index */
        idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

        /* Ensure only 4 MSB bits of mantissa are used as indexes */
        idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
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

        /* Convert input from sf vector to qf32 vector for Horner's method*/
        input_v_qf32 = Q6_Vqf32_vadd_VsfVsf(sline, zero_v_sf);

        /* Perform evaluation of polynomial using Horner's method */
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c3_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

        // x * sigmod
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(input_v_qf32, output_v);

        /* Convert from qf32 to sf, store output and go to handle leftover */
        *((HVX_UVector *)(output_v_ptr++)) = Q6_Vsf_equals_Vqf32(output_v);

        slinep = slinec;
    }

    /* Handle leftover elements */
    if (leftover > 0)
    {
        slinec = (is_in_one_chunk(input_v_ptr, leftover_size, VLEN)
                   ? slinep
                   : *input_v_ptr++);

        sline = Q6_V_valign_VVR(slinec, slinep, (size_t) input);

        /* Shift input range from [input_min, input_max] to [0, input_max - input_min] */
        input_shifted_v_qf32 = Q6_Vqf32_vsub_VsfVsf(sline, input_min_v_f);

        /* Scale shifted input range from [0, input_max - input_min] to [0,16.0) */
        input_scaled_v_qf32 = Q6_Vqf32_vmpy_Vqf32Vqf32(input_shifted_v_qf32, scale_v);

        /*
         * VLUT 16 requires integer indexes. Shift scaled input range from [0,16.0)
         * to [16.0,32.0) in order to convert float indexes to integer values.
         * Float values, represented in IEEE 754, in range [16.0,32.0] have the
         * same exponent, which means 4 MSB of mantissa carry information about
         * integer index.
         */
        input_scaled_v_qf32 = Q6_Vqf32_vadd_Vqf32Vsf(input_scaled_v_qf32, const16_0_v_sf);

        /* Convert back from qf32 to sf in order to extract integer index */
        tmp_v = Q6_Vsf_equals_Vqf32(input_scaled_v_qf32);

        /* Only 4 MSB bits of mantissa represent segment index */
        idx1_v = Q6_Vuw_vlsr_VuwR(tmp_v, 19);

        /* Ensure only 4 MSB bits of mantissa are used as indexes */
        idx1_v = Q6_V_vand_VV(idx1_v, mask_idx1_v);
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

        /* Convert input from sf vector to qf32 vector for Horner's method*/
        input_v_qf32 = Q6_Vqf32_vadd_VsfVsf(sline, zero_v_sf);

        /* Perform evaluation of polynomial using Horner's method */
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(Q6_V_lo_W(c4_coeff_vp), input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c3_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c2_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c1_coeff_vp));
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(output_v, input_v_qf32);
        output_v = Q6_Vqf32_vadd_Vqf32Vqf32(output_v, Q6_V_lo_W(c0_coeff_vp));

        // x * sigmod
        output_v = Q6_Vqf32_vmpy_Vqf32Vqf32(input_v_qf32, output_v);

        /* Convert from qf32 to sf */
        sout = Q6_Vsf_equals_Vqf32(output_v);

        /* Store output */
        vstu_variable(output_v_ptr, leftover_size, sout);
    }

    return 0;
}

#endif

template<typename TensorType>
GraphStatus siluImpl(TensorType& out_0,
                     const TensorType& in_0)

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

#ifdef REFERENCE_OP
   debuglog("silu execute... inval=(%d)", in_0.get_dtype()); 
   debuglog("silu execute... inval=(%d)", out_0.get_dtype()); 
   
    out_0.set_dims(in_0);
    // NHWC

    auto [b_in, h_in, w_in, d_in] = in_0.dims();
    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          // SiLU
          for (Idx d = 0; d < d_in; d++) {
            float inval       = in_0(b, h, w, d);
            float outval      = 1 / (1 + expf(-inval));


            debuglog("silu execute... inval=(%f)", inval);
            debuglog("silu execute... outval=(%f)", outval);

            out_0(b, h, w, d) = inval * outval;
            
          }
        }
      }
    }

#else

    // HVX Method -- FP32 Version
    out_0.set_dims(in_0);
    
    // NHWC
    auto in_ptr = (float*)in_0.raw_data_const();
    auto out_ptr = (float*)out_0.raw_data();
    auto [b_in, h_in, w_in, d_in] = in_0.dims();


    size_t size = b_in*h_in*w_in*d_in;
    
    // Noticable size >= 128
            
    // SiLU  inval / (1 + expf(-inval));
    // sigmod 1.0/(exp(-x)+1.0)
    // SiLU   inval * sigmod
    hvx_silu_af(in_ptr, out_ptr, size);

    
#endif

#ifdef DEBUG

    for (Idx b = 0; b < b_in; b++) {
      for (Idx h = 0; h < h_in; h++) {
        for (Idx w = 0; w < w_in; w++) {
          // SiLU
          for (Idx d = 0; d < d_in; d++) {
            float out_value = out_0(b, h, w, d);
            debuglog("silu execute... outval=(%f)", out_value);
            
          }
        }
      }
    }
  
#endif



  return GraphStatus::Success;
}

__attribute__((unused)) static float siluCostFunc(const Op *op)
{
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}





/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_SiLU);