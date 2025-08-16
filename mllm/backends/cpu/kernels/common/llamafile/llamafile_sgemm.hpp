// Copyright (c) MLLM Team.
// Licensed under the MIT License.
//
// Created by Rongjie Yi on 24-07-23.
//

#pragma once

#include <cstdint>

#include "mllm/core/DataTypes.hpp"

namespace mllm::cpu {

bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void* A, int64_t lda, const void* B, int64_t ldb, void* C,
                     int64_t ldc, int ith, int nth, DataTypes Atype, DataTypes Btype, DataTypes Ctype, void* bias = nullptr,
                     DataTypes BiasType = kFloat32);

bool check_llamafile_sgemm(int64_t m, int64_t n, int64_t k, DataTypes Atype, DataTypes Btype, DataTypes Ctype, int64_t lda,
                           int64_t ldb, int64_t ldc);

}  // namespace mllm::cpu
