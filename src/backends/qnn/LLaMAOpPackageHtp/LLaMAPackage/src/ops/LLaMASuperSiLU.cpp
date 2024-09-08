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
GraphStatus llamasupersiluImpl(TensorType& out_0,
                               const TensorType& in_0,
                               const TensorType& in_1,
                               const PlainFloatTensor& a_scale,
                               const PlainFloatTensor& b_scale,
                               const PlainFloatTensor& o_scale);

// forward declaration of sample cost function
static float llamasupersiluCostFunc(const Op *op);

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
DEF_PACKAGE_PARAM_ORDER("LLaMASuperSiLU", 
                        "a_scale",
                        true,
                        nullptr,
                        "b_scale",
                        true,
                        nullptr,
                        "o_scale",
                        true,
                        nullptr)


/* execute functions for ops */

// #ifndef REFERENCE_OP



// #else

template<typename TensorType>
GraphStatus llamasupersiluImpl(TensorType& out_0,
                               const TensorType& in_0,
                               const TensorType& in_1,
                               const PlainFloatTensor& a_scale,
                               const PlainFloatTensor& b_scale,
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

  float a_scale_ = a_scale(0,0,0,0);
  float b_scale_ = b_scale(0,0,0,0);
  float o_scale_ = o_scale(0,0,0,0);

  auto in_ptr = (uint8_t*)in_0.raw_data_const();
  auto in_ptr2 = (uint8_t*)in_1.raw_data_const();

  auto out_ptr = (uint8_t*)out_0.raw_data();


  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // mul
        for (Idx d = 0; d < d_in; d++) {


          int32_t a_inval       = static_cast<int32_t>(*in_ptr++);
          float a_inval_fp16 = (a_inval-128) * a_scale_;


          int32_t b_inval       = static_cast<int32_t>(*in_ptr2++);
          float b_inval_fp16 = (b_inval-128) * b_scale_;

          
          a_inval_fp16 = a_inval_fp16 * (1 / (1 + expf(-a_inval_fp16)));

          float inval = a_inval_fp16 * b_inval_fp16;
          
          long v = lroundf(inval / o_scale_);
                
          if (v > 127)
              v = 127;
          
          if (v < -128)
              v = -128;
          
          v += 128;

          *out_ptr++ = static_cast<uint8_t>(v);

        }
      }
    }
  }


  return GraphStatus::Success;
}

// #endif

__attribute__((unused)) static float llamasupersiluCostFunc(const Op *op)
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
END_PKG_OP_DEFINITION(PKG_LLaMASuperSiLU);