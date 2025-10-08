#include <string>

#include "mllm/mllm.hpp"

using namespace mllm;  // NOLINT

int main() {
  mllm::initializeContext();
  mllm::setRandomSeed(42);
  for (int i = 0; i < 10; i++) { print(mllm::getRandomState()); }
  mllm::memoryReport();
}
