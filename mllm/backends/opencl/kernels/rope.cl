#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void rope_f32(__global const float *src, __global float *dst,
                       __global const float *sin_table,
                       __global const float *cos_table, const int H,
                       const int S, const int D) {
  int d_half = get_global_id(0);
  int s = get_global_id(1);
  int bh = get_global_id(2);

  if (d_half >= D / 2)
    return;

  int b = bh / H;
  int h = bh % H;

  int d = d_half;
  int d_pair = d + D / 2;

  // Input shape: [B, H, S, D]
  int in_idx1 = b * (H * S * D) + h * (S * D) + s * D + d;
  int in_idx2 = b * (H * S * D) + h * (S * D) + s * D + d_pair;

  // Sin/Cos shape: [B, S, D]
  int sc_idx1 = b * (S * D) + s * D + d;

  float v1 = src[in_idx1];
  float v2 = src[in_idx2];

  float s1 = sin_table[sc_idx1];
  float c1 = cos_table[sc_idx1];

  float out1 = v1 * c1 - v2 * s1;
  float out2 = v1 * s1 + v2 * c1;

  dst[in_idx1] = out1;
  dst[in_idx2] = out2;
}

__kernel void rope_f16(__global const half *src, __global half *dst,
                       __global const float *sin_table,
                       __global const float *cos_table, const int H,
                       const int S, const int D) {
  int d_half = get_global_id(0);
  int s = get_global_id(1);
  int bh = get_global_id(2);

  if (d_half >= D / 2)
    return;

  int b = bh / H;
  int h = bh % H;

  int d = d_half;
  int d_pair = d + D / 2;

  // Input shape: [B, H, S, D]
  int in_idx1 = b * (H * S * D) + h * (S * D) + s * D + d;
  int in_idx2 = b * (H * S * D) + h * (S * D) + s * D + d_pair;

  // Sin/Cos shape: [B, S, D]
  int sc_idx1 = b * (S * D) + s * D + d;

  float v1 = (float)src[in_idx1];
  float v2 = (float)src[in_idx2];

  float s1 = sin_table[sc_idx1];
  float c1 = cos_table[sc_idx1];

  float out1 = v1 * c1 - v2 * s1;
  float out2 = v1 * s1 + v2 * c1;

  dst[in_idx1] = (half)out1;
  dst[in_idx2] = (half)out2;
}
