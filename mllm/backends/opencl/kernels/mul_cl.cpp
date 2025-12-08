#include "a_opencl_source_map.hpp"
namespace mllm::opencl {
const char* mul = "__kernel void mul_float(__global const float *A,__global const float *B,\n"
                  " __global float *C) {\n"
                  " size_t index=get_global_id(0);\n"
                  " C[index]=A[index]*B[index];\n"
                  "}\n";
}  // namespace mllm::opencl
