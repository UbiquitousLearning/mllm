//==============================================================================
// Auto Generated Code for LLaMAPackage
//==============================================================================

#include "HTP/core/constraints.h"
#include "HTP/core/op_package_feature_support.h"
#include "HTP/core/op_register_ext.h"
#include "HTP/core/optimize.h"
#include "QnnOpPackage.h"
#include "HTP/core/simple_reg.h"

BEGIN_PKG_OP_DEFINITION(PKG_LLaMALinear);

// op execute function declarations
template<typename TensorType>
GraphStatus llamalinearImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1, const TensorType& in_2,
                            const PlainFloatTensor& in_scale, const PlainFloatTensor& weight_scale,
                            const PlainFloatTensor& bias_scale, const PlainFloatTensor& output_scale);

// forward declaration of sample cost function
static float llamalinearCostFunc(const Op* op);

/*
 * method 1 for defining op, using default cost value (i.e. GLACIAL) and default flag (Flags::RESOURCE_HVX)
 * syntax: DEF_PACKAGE_OP(F,OP)
 * e.g. DEF_PACKAGE_OP((llamalinearImpl<Tensor>), "LLaMALinear")
 */
DEF_PACKAGE_OP((llamalinearImpl<Tensor>), "LLaMALinear")

/*
 * method 2 for defining op with specified cost value (one of GLACIAL, SNAIL, FAST, FREE)
 * and provided flags
 * syntax: DEF_PACKAGE_OP_AND_COST_AND_FLAGS(F,OP,COST,...)
 * can use zero or more flags, FLAG options are IS_CONST, INHIBIT_CONST_PROP,
 * RESOURCE_HVX, RESOURCE_HMX(not supported in external op packages)
 * e.g. DEF_PACKAGE_OP_AND_COST_AND_FLAGS((llamalinearImpl<PlainFloatTensor>), "LLaMALinear", SNAIL)
 */

/*
 * method 3 for defining op with cost function pointer and provided flags
 * cost function pointer type: typedef float (*cost_function) (const Op * op);
 * syntax: DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS(F,OP,COST_F,...)
 * e.g. DEF_PACKAGE_OP_AND_COST_F_AND_FLAGS((llamalinearImpl<PlainFloatTensor>),
 * "LLaMALinear", llamalinearCostFunc, Flags::RESOURCE_HVX)
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
DEF_PACKAGE_PARAM_ORDER("LLaMALinear", "in_scale", true, nullptr, "weight_scale", true, nullptr, "bias_scale", true, nullptr,
                        "output_scale", true, nullptr)

/* execute functions for ops */

float Round(float num) {
  float floor_num = floor(num);
  float ceil_num = ceil(num);

  if (num - floor_num < ceil_num - num) {
    return floor_num;
  } else {
    return ceil_num;
  }
}

template<typename TensorType>
GraphStatus llamalinearImpl(TensorType& out_0, const TensorType& in_0, const TensorType& in_1, const TensorType& in_2,
                            const PlainFloatTensor& in_scale, const PlainFloatTensor& weight_scale,
                            const PlainFloatTensor& bias_scale, const PlainFloatTensor& output_scale)

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
  //  假设输入张量是4维的，NHWC格式
  int batch_size = in_0.dims()[0];
  int height = in_0.dims()[1];
  int width = in_0.dims()[2];
  int in_features = in_0.dims()[3];   // 输入的通道数
  int out_features = in_1.dims()[3];  // 输出的特征数（即输出通道数）

  // 检查输入张量的形状是否匹配
  if (in_1.dims()[0] != 1 || in_1.dims()[1] != 1 || in_1.dims()[2] != in_features || in_2.dims()[3] != out_features) {
    return GraphStatus::ErrorFatal;
  }

  // 获取量化比例
  float w_scale = weight_scale(0, 0, 0, 0);
  float i_scale = in_scale(0, 0, 0, 0);
  float b_scale = bias_scale(0, 0, 0, 0);
  float o_scale = output_scale(0, 0, 0, 0);

  // 初始化输出张量

  size_t dims[] = {static_cast<size_t>(batch_size), static_cast<size_t>(height), static_cast<size_t>(width),
                   static_cast<size_t>(out_features)};
  out_0.set_dims(dims);

  // only support float bias now.
  auto in0_ptr = (uint8_t*)in_0.raw_data_const();
  auto in1_ptr = (uint8_t*)in_1.raw_data_const();
  auto in2_ptr = (uint8_t*)in_2.raw_data_const();
  auto out_ptr = (int8_t*)out_0.raw_data();

  // 进行量化Linear乘法
  for (int b = 0; b < batch_size; ++b) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        for (int n = 0; n < out_features; ++n) {
          float acc = 0;
          for (int k = 0; k < in_features; ++k) {
            int in_index = b * height * width * in_features + h * width * in_features + w * in_features + k;
            int weight_index = k * out_features + n;
            acc += ((static_cast<int32_t>(in0_ptr[in_index]) - 128) * i_scale)
                   * ((static_cast<int32_t>(in1_ptr[weight_index]) - 128) * w_scale);
          }
          // 加上偏置并进行反量化
          float result = acc;
          result += (static_cast<int32_t>(in2_ptr[n]) - 128) * b_scale;
          // 将结果限制在uint8范围内
          int out_index = b * height * width * out_features + h * width * out_features + w * out_features + n;

          result = Round(result / o_scale);

          long v = lroundf(result);

          if (v > 127) v = 127;

          if (v < -128) v = -128;

          if (out_0.get_dtype() == DType::QUInt8) v += 128;

          out_ptr[out_index] = static_cast<uint8_t>(v);
        }
      }
    }
  }

  return GraphStatus::Success;
}

__attribute__((unused)) static float llamalinearCostFunc(const Op* op) {
  /*
   * add code here
   * */

  float cost = 0.0;  // add cost computation here
  return cost;
}

/* At the bottom of the op file, call END_PKG_OP_DEFINITION(<name>),
   where <name> is as BEGIN_PKG_OP_DEFINITION
*/
END_PKG_OP_DEFINITION(PKG_LLaMALinear);