#ifndef MLLM_TRACY_HPP
#define MLLM_TRACY_HPP

#ifdef MLLM_TRACY_ENABLE
#include "tracy/Tracy.hpp"
#define MLLM_TRACY_ZONE_SCOPED ZoneScoped
#define MLLM_TRACY_ZONE_SCOPED_NAMED(name) ZoneScopedN(name)
#define MLLM_TRACY_FRAME_MARK FrameMark
#else
#define MLLM_TRACY_ZONE_SCOPED
#define MLLM_TRACY_ZONE_SCOPED_NAMED(name)
#define MLLM_TRACY_FRAME_MARK
#endif

#endif  // MLLM_TRACY_HPP