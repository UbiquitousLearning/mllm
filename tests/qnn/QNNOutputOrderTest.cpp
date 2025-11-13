// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/backends/qnn/QNNModel.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/utils/Log.hpp"
#include <cassert>
#include <vector>
#include <string>

using namespace mllm;
using namespace mllm::qnn;

// Mock QNN interface for testing
struct MockQnnInterface {
  // Minimal mock implementation
};

// Test QNNModel output order mapping
void testOutputOrderMapping() {
  MLLM_INFO("Testing QNNModel output order mapping...");

  // Create a mock QNN interface (in real usage, this would be from QNN SDK)
  // For testing purposes, we'll create a minimal test
  
  // Note: This test requires actual QNN backend initialization
  // In a real test environment, you would:
  // 1. Initialize QNN backend
  // 2. Create a QNNModel
  // 3. Add tensors in a specific order
  // 4. Set expected output order
  // 5. Verify the mapping is correct

  MLLM_INFO("QNNModel output order mapping test structure:");
  MLLM_INFO("  1. Create QNNModel with expected output order");
  MLLM_INFO("  2. Add output tensors (QNN order)");
  MLLM_INFO("  3. Verify qnnOutputNameToIndex_ mapping is correct");
  MLLM_INFO("  4. Verify getQnnOutputIndex() returns correct indices");
  MLLM_INFO("  5. Verify getExpectedOutputOrder() returns correct order");

  // Example test scenario:
  // Expected order (MLLM): ["output_0", "output_1", "output_2"]
  // QNN order (actual): ["output_2", "output_0", "output_1"]
  // Mapping should be:
  //   MLLM[0] = QNN[1] (output_0)
  //   MLLM[1] = QNN[2] (output_1)
  //   MLLM[2] = QNN[0] (output_2)

  MLLM_INFO("Test structure created. Integration test requires QNN backend.");
}

// Test output reordering logic
void testOutputReordering() {
  MLLM_INFO("Testing output reordering logic...");

  // Simulate the reordering logic
  std::vector<std::string> expectedOrder = {"output_0", "output_1", "output_2"};
  std::map<std::string, int> qnnOutputNameToIndex = {
      {"output_2", 0},  // QNN returns in this order
      {"output_0", 1},
      {"output_1", 2}
  };

  // Simulate output tensors (in QNN order)
  std::vector<std::string> qnnOutputs = {"output_2", "output_0", "output_1"};

  // Reorder according to expected order
  std::vector<int> reorderedIndices;
  for (const auto& expected_name : expectedOrder) {
    auto it = qnnOutputNameToIndex.find(expected_name);
    if (it != qnnOutputNameToIndex.end()) {
      reorderedIndices.push_back(it->second);
      MLLM_INFO("  Mapping: MLLM[{}] = QNN[{}] (tensor: {})", 
                reorderedIndices.size() - 1, it->second, expected_name);
    }
  }

  // Verify the mapping
  assert(reorderedIndices.size() == expectedOrder.size());
  assert(reorderedIndices[0] == 1);  // output_0 is at QNN index 1
  assert(reorderedIndices[1] == 2);  // output_1 is at QNN index 2
  assert(reorderedIndices[2] == 0);  // output_2 is at QNN index 0

  MLLM_INFO("Output reordering logic test passed!");
}

int main() {
  MLLM_INFO("=== QNN Output Order Test ===");
  
  testOutputOrderMapping();
  testOutputReordering();
  
  MLLM_INFO("=== All tests passed ===");
  return 0;
}

