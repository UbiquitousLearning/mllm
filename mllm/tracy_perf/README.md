# Tracy Profiler Integration

This document provides instructions on how to use the Tracy profiler within the MLLM project.

## Enabling Tracy

To enable Tracy, you need to set the `MLLM_TRACY_ENABLE` option to `ON` when configuring the project with CMake:

```bash
cmake -DMLLM_TRACY_ENABLE=ON ..
```

## API Usage

The following macros are available for profiling your code:

- `MLLM_TRACY_ZONE_SCOPED`: Creates a new profiling zone that is automatically destroyed when the scope is exited.
- `MLLM_TRACY_ZONE_SCOPED_NAMED(name)`: Creates a new profiling zone with a custom name.
- `MLLM_TRACY_FRAME_MARK`: Marks the end of a frame.

### Example

```cpp
#include "tracy_perf/Tracy.hpp"

void myFunction() {
    MLLM_TRACY_ZONE_SCOPED;
    // Your code here
}

void anotherFunction() {
    MLLM_TRACY_ZONE_SCOPED_NAMED("My Custom Zone");
    // Your code here
}

int main() {
    while (true) {
        myFunction();
        anotherFunction();
        MLLM_TRACY_FRAME_MARK;
    }
    return 0;
}
```