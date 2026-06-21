#include "a_opencl_source_map.hpp"
namespace mllm::opencl {
const char* fill = "__kernel void fill_fp32(float value,__global float *dst) {\n"
                   " size_t index=get_global_id(0);\n"
                   " dst[index]=value;\n"
                   "}\n"
                   "__kernel void fill_arange_fp32(float start,float step,__global float *dst) {\n"
                   " size_t index=get_global_id(0);\n"
                   " dst[index]=start+(float)index*step;\n"
                   "}\n";
}  // namespace mllm::opencl
