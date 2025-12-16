#include <iostream>
#include <vector>
#include <cstring>
#include <acl/acl.h>
#include "mllm/mllm.hpp"
#include "mllm/backends/ascend/AscendCommon.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/engine/Context.hpp"
#include "mllm/core/aops/ElewiseOps.hpp"
#include "mllm/core/OpTypes.hpp"

using namespace mllm;

int main() {
  std::cout << "=== Ascend Add Op Demo ===" << std::endl;

  try {
    std::cout << "1. Initializing Ascend backend..." << std::endl;
    initAscendBackend();
    std::cout << "   ✓ Ascend backend initialized\n" << std::endl;

    std::cout << "2. Preparing test data..." << std::endl;
    const int batch = 2;
    const int size = 3;
    std::vector<float> data_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::vector<float> data_y = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    std::vector<float> expected = {11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f};
    
    std::cout << "   Input X: [";
    for (size_t i = 0; i < data_x.size(); ++i) {
      std::cout << data_x[i];
      if (i < data_x.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "   Input Y: [";
    for (size_t i = 0; i < data_y.size(); ++i) {
      std::cout << data_y[i];
      if (i < data_y.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n" << std::endl;

    std::cout << "3. Preparing tensors on Ascend..." << std::endl;
    auto x_handle = ascend::prepareAscendTensor(data_x, batch, size);
    auto y_handle = ascend::prepareAscendTensor(data_y, batch, size);
    auto& x_ascend = x_handle.tensor();
    auto& y_ascend = y_handle.tensor();
    std::cout << "   ✓ Tensors ready on Ascend device\n" << std::endl;

    std::cout << "4. Executing Add operation on Ascend..." << std::endl;
    auto& ctx = Context::instance();
    std::cout << "context over" <<std::endl;
    auto z_ascend = ctx.buildOpAndSubmitTask(
        OpTypes::kAdd, 
        aops::AddOpOptions{}, 
        {x_ascend, y_ascend}
    )[0];
    std::cout << "   ✓ Add operation completed\n" << std::endl;

  std::cout << "\n5. Copying result from NPU to CPU for verification..." << std::endl;
  std::vector<float> actual;
  bool correct = ascend::verifyAscendTensor(
      z_ascend,
      expected,
      /*atol=*/1e-2f,
      /*rtol=*/1e-2f,
      /*verbose=*/true,
      &actual);

  std::cout << "   Actual result:   [";
  for (size_t i = 0; i < actual.size(); ++i) {
    std::cout << actual[i];
    if (i < actual.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  std::cout << "   Expected result: [";
  for (size_t i = 0; i < expected.size(); ++i) {
    std::cout << expected[i];
    if (i < expected.size() - 1) std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  if (correct) {
    std::cout << "\n✓✓✓ Test PASSED! All values match expected results. ✓✓✓" << std::endl;
  } else {
    std::cout << "\n✗✗✗ Test FAILED! Results don't match expected values. ✗✗✗" << std::endl;
  }
  
    x_handle.release();
    y_handle.release();
    
    return correct ? 0 : 1;

  } catch (const std::exception& e) {
    std::cerr << "\n✗ Error: " << e.what() << std::endl;
    return 1;
  }
}

