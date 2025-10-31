#include <thread>
#include <chrono>
#include "mllm/tracy_perf/Tracy.hpp"

void myFunction() {
  MLLM_TRACY_ZONE_SCOPED;
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void anotherFunction() {
  MLLM_TRACY_ZONE_SCOPED_NAMED("My Custom Zone");
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
}

int main() {
  int i = 0;
  while (i < 100) {
    myFunction();
    anotherFunction();
    MLLM_TRACY_FRAME_MARK;
    i++;
  }
  return 0;
}
