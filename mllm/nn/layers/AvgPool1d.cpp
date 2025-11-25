#include "mllm/nn/layers/AvgPool1d.hpp"
#include "mllm/core/aops/AvgPool1dOp.hpp"
#include "mllm/nn/Layer.hpp"

namespace mllm::nn {

AvgPool1d::AvgPool1d() : Layer(OpTypes::kAvgPool1d, aops::AvgPool1dOpOptions{}) {}

AvgPool1d::AvgPool1d(int32_t kernel_size, int32_t stride, int32_t padding, bool ceil_mode, bool count_include_pad)
    : Layer(OpTypes::kAvgPool1d, aops::AvgPool1dOpOptions{.kernel_size = kernel_size,
                                                           .stride = stride,
                                                           .padding = padding,
                                                           .ceil_mode = ceil_mode,
                                                           .count_include_pad = count_include_pad}) {}

AvgPool1d::AvgPool1d(const aops::AvgPool1dOpOptions& options) : Layer(OpTypes::kAvgPool1d, options) {}

}  // namespace mllm::nn
