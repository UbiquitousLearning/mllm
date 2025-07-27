#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void like(
    __global void *output,
    const float like_value,
    const int count,
    const int dtype_size) { // sizeof(float) or sizeof(half)

    const int gid = get_global_id(0);
    if (gid >= count) {
        return;
    }

    if (dtype_size == 4) { // float
        __global float *out_fp32 = (__global float *)output;
        out_fp32[gid] = like_value;
    } else if (dtype_size == 2) { // half
        __global half *out_fp16 = (__global half *)output;
        out_fp16[gid] = (half)like_value;
    }
}