// Copyright (c) MLLM Team.
// Licensed under the MIT License.

// Inspired by torchao's valpack.

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace mllm::cpu::arm::bitspack {

void interleave_data(void* data_interleaved, const void* data, int bytes_per_val, int vals_per_channel, int vals_per_group,
                     int vals_per_chunk, int channels, int channel_stride_in_vals);

}
