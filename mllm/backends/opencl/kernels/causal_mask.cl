#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void causal_mask_fp32(__global const float *input,
                               __global float *output, const int B, const int H,
                               const int S, const int D,
                               const int sliding_window,
                               const int window_size) {
  const int d = get_global_id(0);
  const int s = get_global_id(1);
  const int bh = get_global_id(2);

  if (d >= D || s >= S || bh >= B * H) {
    return;
  }

  const int offset = (bh * S + s) * D + d;
  float val = input[offset];

  if (S == 1) {
    output[offset] = val;
    return;
  }

  if (sliding_window == 0) {
    // Standard causal mask
    int copy_count = D - S + s + 1;
    if (d >= copy_count) {
      val = -1e10f;
    }
  } else {
    // Sliding window
    int copy_start_idx = s - window_size + 1;
    if (copy_start_idx < 0)
      copy_start_idx = 0;

    int copy_end_idx = s + 1;

    if (d < copy_start_idx || d >= copy_end_idx) {
      val = -1e10f;
    }
  }

  output[offset] = val;
}

__kernel void causal_mask_fp16(__global const half *input,
                               __global half *output, const int B, const int H,
                               const int S, const int D,
                               const int sliding_window,
                               const int window_size) {
  const int d = get_global_id(0);
  const int s = get_global_id(1);
  const int bh = get_global_id(2);

  if (d >= D || s >= S || bh >= B * H) {
    return;
  }

  const int offset = (bh * S + s) * D + d;
  half val = input[offset];

  if (S == 1) {
    output[offset] = val;
    return;
  }

  if (sliding_window == 0) {
    int copy_count = D - S + s + 1;
    if (d >= copy_count) {
      val = -65500.0h;
    }
  } else {
    int copy_start_idx = s - window_size + 1;
    if (copy_start_idx < 0)
      copy_start_idx = 0;

    int copy_end_idx = s + 1;

    if (d < copy_start_idx || d >= copy_end_idx) {
      val = -65500.0h;
    }
  }

  output[offset] = val;
}
