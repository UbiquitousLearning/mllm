// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

namespace mllm::ffi {

class MllmAny;

class MllmAnyView {
 protected:
  friend class MllmAny;
};

}  // namespace mllm::ffi
