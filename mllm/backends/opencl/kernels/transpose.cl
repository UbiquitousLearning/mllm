__kernel void transpose_generic(__global const float *input,
                                __global float *output,
                                __global const int *in_strides,
                                __global const int *out_shape,
                                __global const int *perm, const int ndim,
                                const int total_elements) {
  int global_id = get_global_id(0);
  if (global_id >= total_elements)
    return;

  int temp_idx = global_id;
  int in_idx = 0;

  for (int i = 0; i < ndim; ++i) {
    int d = ndim - 1 - i;
    int coord = temp_idx % out_shape[d];
    temp_idx /= out_shape[d];
    in_idx += coord * in_strides[perm[d]];
  }

  output[global_id] = input[in_idx];
}

__kernel void transpose_0213(__global const float *input,
                             __global float *output, const int B, const int S,
                             const int H, const int D) {
  int global_id = get_global_id(0);
  int total = B * S * H * D;
  if (global_id >= total)
    return;

  int d = global_id % D;
  int temp = global_id / D;
  int s = temp % S;
  temp = temp / S;
  int h = temp % H;
  int b = temp / H;

  int in_idx = ((b * S + s) * H + h) * D + d;

  output[global_id] = input[in_idx];
}

__kernel void transpose_0132(__global const float *input,
                             __global float *output, const int B, const int C,
                             const int H, const int W) {
  int global_id = get_global_id(0);
  int total = B * C * H * W;
  if (global_id >= total)
    return;

  int h = global_id % H;
  int temp = global_id / H;
  int w = temp % W;
  temp = temp / W;
  int c = temp % C;
  int b = temp / C;

  int in_idx = ((b * C + c) * H + h) * W + w;

  output[global_id] = input[in_idx];
}
