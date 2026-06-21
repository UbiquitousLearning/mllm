__kernel void fill_fp32(float value, __global float *dst) {
  size_t index = get_global_id(0);
  dst[index] = value;
}

__kernel void fill_arange_fp32(float start, float step, __global float *dst) {
  size_t index = get_global_id(0);
  dst[index] = start + (float)index * step;
}
