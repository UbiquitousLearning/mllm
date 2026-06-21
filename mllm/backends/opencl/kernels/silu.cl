#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void silu_fp32(__global const float *input, __global float *output) {
  const int i = get_global_id(0);
  const float x = input[i];

  output[i] = x / (1.0f + exp(-x));
}

__kernel void silu_fp16(__global const half *input, __global half *output,
                        const int count) {

  const int vec_idx = get_global_id(0);
  const int vec_limit = count / 4;

  if (vec_idx < vec_limit) {
    const int i = vec_idx * 4;
    half4 x = vload4(0, input + i);
    half4 result = x / ((half4)(1.0h) + exp(-x));
    vstore4(result, 0, output + i);
  }

  const int remainder_start = vec_limit * 4;
  if (get_local_id(0) < (count - remainder_start) && get_group_id(0) == 0) {
    const int i = remainder_start + get_local_id(0);
    if (i < count) {
      const half x = input[i];
      output[i] = x / (1.0h + exp(-x));
    }
  }
}