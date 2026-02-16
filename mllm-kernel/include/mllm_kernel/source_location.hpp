// Copyright SGLang Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include <version>

/// NOTE: fallback to a minimal source_location implementation
#if defined(__cpp_lib_source_location)
#include <source_location>

using source_location_t = std::source_location;

#else

struct source_location_fallback {
 public:
  static constexpr source_location_fallback current() noexcept { return source_location_fallback{}; }
  constexpr source_location_fallback() noexcept = default;
  constexpr unsigned line() const noexcept { return 0; }
  constexpr unsigned column() const noexcept { return 0; }
  constexpr const char* file_name() const noexcept { return ""; }
  constexpr const char* function_name() const noexcept { return ""; }
};

using source_location_t = source_location_fallback;

#endif
