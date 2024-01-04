//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"


BEGIN_PKG_OP_DEFINITION(PKG_RoPE);


// op execute function declarations
template<typename TensorType>
GraphStatus ropeImpl(TensorType& out_0,
                     const TensorType& in_0,
                     const TensorType& sin,
                     const TensorType& cos,
                     const Tensor& pose_type);

// forward declaration of sample cost function
static float ropeCostFunc(const Op *op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((ropeImpl<Tensor>), "RoPE")
 */
DEF_PACKAGE_OP((ropeImpl<Tensor>), "RoPE")

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
DEF_PACKAGE_PARAM_ORDER("RoPE", 
                        "pose_type",
                        true,
                        nullptr)


/* execute functions for ops */

template<typename TensorType>
GraphStatus ropeImpl(TensorType& out_0,
                     const TensorType& in_0,
                     const TensorType& sin,
                     const TensorType& cos,
                     const Tensor& pose_type)

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

   debuglog("relux execute... dims=(%zdx%zdx%zdx%zd)", in_0.dim(0), in_0.dim(1), in_0.dim(2), in_0.dim(3));
   debuglog("relux execute... dims=(%zdx%zdx%zdx%zd)", sin.dim(0), sin.dim(1), sin.dim(2), sin.dim(3));
   debuglog("relux execute... dims=(%zdx%zdx%zdx%zd)", cos.dim(0), cos.dim(1), cos.dim(2), cos.dim(3));

  // BSHD =>  NHWC

  int h_cnt_ = 0; // history sequence position

  // Todo: We need consider to store the sequence position if we have KV Cache

  auto pose_type_ = pose_type(0,0,0,0);

  out_0.set_dims(in_0);
  auto [b_in, h_in, w_in, d_in] = in_0.dims();
  for (Idx b = 0; b < b_in; b++) {
    for (Idx h = 0; h < h_in; h++) {
      for (Idx w = 0; w < w_in; w++) {
        // RoPE
        for (Idx d = 0; d < d_in; d++) {
          

          int s = h; //  BSHD order
          if (pose_type_ == 1) {
              float in_value = in_0(b, h, w, d);
              float in_value_2;
              if (d < d_in / 2) { // 偶數 0,2,4
                  in_value_2 = -in_0(b, h, w, d + d_in / 2);
              } else {
                  in_value_2 = in_0(b, h, w, d - d_in / 2);
              }
              float sin_value = sin(0, 0, s +h_cnt_, d);
              float cos_value = cos(0, 0, s +h_cnt_, d);
              auto value = in_value * cos_value + in_value_2 * sin_value;
              out_0(b, h, w, d) = value;
          }
          else if (pose_type_ == 2) {
              float in_value = in_0(b, h, w, d);
              float in_value_2;
              if (d % 2 == 0) { // 偶數 0,2,4
                  in_value_2 = -in_0(b, h, w, d + 1);
              } else {
                  in_value_2 = in_0(b, h, w, d - 1);
              }
              float sin_value = sin(0, 0, s +h_cnt_, d);
              float cos_value = cos(0, 0, s +h_cnt_, d);
              auto value = in_value * cos_value + in_value_2 * sin_value;
              out_0(b, h, w, d) = value;
          } else {
              float in_value = in_0(b, h, w, d);
              float in_value_2;
              float sin_value = sin(0, 0, s +h_cnt_, d);
              float cos_value = cos(0, 0, s +h_cnt_, d);
              if (d < d_in / 4) {
                  in_value_2 = -in_0(b, h, w, d + d_in / 4);
                  auto value = in_value * cos_value + in_value_2 * sin_value;

                  out_0(b ,h , w, d) = value;
              } else if(d < d_in / 2){
                  in_value_2 = in_0(b, h, w, d - d_in / 4);
                  auto value = in_value * cos_value + in_value_2 * sin_value;
                  
                  out_0(b ,h , w, d) = value;
              }else {
                  
                  out_0(b ,h , w, d) = in_value;
              }
          }

        }
      }
    }
  }


//   auto &input = inputs[0];
//   auto &output = outputs[0];
//   for (int n = 0; n < input->batch(); ++n) {
//       for (int h = 0; h < input->head(); ++h) {
//           for (int s = 0; s < input->sequence(); ++s) {//sequance
//               #pragma omp parallel for num_threads(4)
//               for (int d = 0; d < input->dimension(); ++d) {
//                   if (pose_type_== 1) {
//                       float in_value = input->dataAt<float>(n, h, s, d);
//                       float in_value_2;
//                       if (d < input->dimension() / 2) { // 偶數 0,2,4
//                           in_value_2 = -input->dataAt<float>(n, h, s, d + input->dimension() / 2);
//                       } else {
//                           in_value_2 = input->dataAt<float>(n, h, s, d - input->dimension() / 2);
//                       }
//                       float sin_value = sin_.dataAt<float>(0, 0, s +h_cnt_, d);
//                       float cos_value = cos_.dataAt<float>(0, 0, s +h_cnt_, d);
//                       auto value = in_value * cos_value + in_value_2 * sin_value;
//                       if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
//                           output->setDataAt<float>(n, h, s, d, value);
//                       }
//                       else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
//                           output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
//                       }
//                   }
//                   else if (pose_type_== 2) {
//                       float in_value = input->dataAt<float>(n, h, s, d);
//                       float in_value_2;
//                       if (d % 2 == 0) { // 偶數 0,2,4
//                           in_value_2 = -input->dataAt<float>(n, h, s, d + 1);
//                       } else {
//                           in_value_2 = input->dataAt<float>(n, h, s, d - 1);
//                       }
//                       float sin_value = sin_.dataAt<float>(0, 0, s +h_cnt_, d);
//                       float cos_value = cos_.dataAt<float>(0, 0, s +h_cnt_, d);
//                       auto value = in_value * cos_value + in_value_2 * sin_value;
//                       if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
//                           output->setDataAt<float>(n, h, s, d, value);
//                       }
//                       else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
//                           output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
//                       }
//                   }else{
//                       float in_value = input->dataAt<float>(n, h, s, d);
//                       float in_value_2;
//                       float sin_value = sin_.dataAt<float>(0, 0, s +h_cnt_, d);
//                       float cos_value = cos_.dataAt<float>(0, 0, s +h_cnt_, d);
//                       if (d < input->dimension() / 4) {
//                           in_value_2 = - input->dataAt<float>(n, h, s, d + input->dimension() / 4);
//                           auto value = in_value * cos_value + in_value_2 * sin_value;
//                           if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
//                               output->setDataAt<float>(n, h, s, d, value);
//                           }
//                           else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
//                               output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
//                           }
//                       } else if(d < input->dimension() / 2){
//                           in_value_2 = input->dataAt<float>(n, h, s, d - input->dimension() / 4);
//                           auto value = in_value * cos_value + in_value_2 * sin_value;
//                           if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
//                               output->setDataAt<float>(n, h, s, d, value);
//                           }
//                           else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
//                               output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(value));
//                           }
//                       }else {
//                           if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F32) {
//                               output->setDataAt<float>(n, h, s, d, in_value);
//                           }
//                           else if(output->dtypeAt(n,h,s, d) == MLLM_TYPE_F16) {
//                               output->setDataAt<mllm_fp16_t>(n, h, s, d, MLLM_FP32_TO_FP16(in_value));
//                           }
//                       }
//                   }
//               }
//           }
//       }
//   }


// Todo store history position
//   h_cnt_ += input->sequence();
//   if(h_cnt_ >pos_max_){
//       h_cnt_ = 0;
//   }


  return GraphStatus::Success;
}

__attribute__((unused)) static float ropeCostFunc(const Op *op)
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
END_PKG_OP_DEFINITION(PKG_RoPE);