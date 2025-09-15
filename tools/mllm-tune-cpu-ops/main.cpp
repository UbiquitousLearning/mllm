// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/experiments/auto_tune/TuningSpace.hpp"

#include "CPUOps.hpp"

int main() {
  mllm::initializeContext();

  // EW FP32 Add
  ElewiseAddFloat32 elewise_add_float32 = ElewiseAddFloat32();
  elewise_add_float32.addTuningParameter("size", {128, 256, 512, 1024});
  elewise_add_float32.addTuningParameter("threads", {1, 2, 4, 8});
  elewise_add_float32.tune();

  mllm::shutdownContext();
}
