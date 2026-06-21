# CPU Kernels Unit Test Guide for mllm

This guide provides comprehensive instructions for AI agents to generate appropriate CPU Kernel Tests for user-specified kernels in the mllm project.

## Overview

The mllm project uses GoogleTest framework for unit testing CPU kernels. Each kernel test validates the correctness of kernel implementations across different data types and tensor shapes.

## Test Structure

### 1. Test Class Hierarchy

All CPU kernel tests inherit from the `KernelTest` base class:

```cpp
class KernelTest : public testing::Test {
 public:
  KernelTest() = default;
  ~KernelTest() override = default;

  void SetUp() override {}
  void TearDown() override {}
};
```

For specific kernel types, intermediate classes are created, such as `ElementwiseKernelTest`.

### 2. Test File Organization

- **KernelTest.cpp**: Main test file containing TEST_F declarations
- **[KernelName]KernelTest.hpp**: Header file with test implementation
- **KernelTestHelper.hpp**: Common helper functions and base classes

## Creating a New Kernel Test

### Step 1: Create Test Header File

Create a new header file for your kernel test, e.g., `MyKernelTest.hpp`:

```cpp
#include "mllm/mllm.hpp"
#include "KernelTestHelper.hpp"

class MyKernelTest : public KernelTest {
 public:
  MyKernelTest() = default;
  ~MyKernelTest() override = default;
  
  // Declare your test functions here
  bool MyKernelFloat32Test(const std::vector<mllm::Tensor::shape_t>& shapes);
};
```

### Step 2: Implement Test Functions

In the header file, implement your test functions:

```cpp
bool MyKernelTest::MyKernelFloat32Test(const std::vector<mllm::Tensor::shape_t>& shapes) {
  using mllm::kCPU;
  using mllm::kFloat32;
  using mllm::Tensor;
  
  for (auto& s : shapes) {
    // Create input tensors with random data
    Tensor a = Tensor::random(s, -3, 3, kFloat32, kCPU);
    Tensor b = Tensor::random(s, -3, 3, kFloat32, kCPU);
    
    // Compute reference result (manual implementation)
    Tensor ref_c = Tensor::zeros(s, kFloat32, kCPU);
    {
      auto a_ptr = a.ptr<mllm_fp32_t>();
      auto b_ptr = b.ptr<mllm_fp32_t>();
      auto c_ptr = ref_c.ptr<mllm_fp32_t>();
      auto num_elements = a.numel();
      for (size_t i = 0; i < num_elements; i++) {
        // Replace with your kernel's reference implementation
        c_ptr[i] = a_ptr[i] + b_ptr[i]; // Example: element-wise addition
      }
    }
    
    // Compute actual result using mllm kernel
    auto c = a + b; // Replace with your kernel operation
    
    // Compare results
    auto result = mllm::test::allClose(c, ref_c);
    if (!result) {
      // Print debug information on failure
      mllm::print(c);
      mllm::print(ref_c);
      mllm::print(result);
      return false;
    }
  }
  return true;
}
```

### Step 3: Register Tests in KernelTest.cpp

Include your test header and register the tests:

```cpp
// Include your test header
#include "MyKernelTest.hpp"

// Register test cases
TEST_F(MyKernelTest, MyKernelFloat32) {
  EXPECT_EQ(MyKernelFloat32Test({
                {42},
                {5, 5},
                {16, 16},
                {16, 18},
                {32, 32},
                {128, 128, 128},
            }),
            true);
}
```

## Test Implementation Guidelines

### 1. Test Data Types

Ensure your kernel test covers all relevant data types:
- kFloat32 (mllm_fp32_t)
- kFloat16 (mllm_fp16_t)
- kInt8 (mllm_int8_t)
- kInt16 (mllm_int16_t)
- kInt32 (mllm_int32_t)

### 2. Test Tensor Shapes

Use a standard set of tensor shapes to validate your kernel:
- 1D tensors: {42}, {5, 5}, {16, 16}, {16, 18}
- 2D tensors: {32, 32}, {128, 128}
- 3D tensors: {128, 128, 128}

### 3. Reference Implementation

Always provide a reference implementation that:
- Is mathematically correct
- Covers edge cases
- Is simple and readable
- Matches the kernel's intended behavior

### 4. Error Handling

On test failure, provide informative output:
- Print the actual and expected tensors
- Show the comparison result
- Include tensor shapes and data types

## Using Macros for Test Generation

For repetitive test patterns, use macros to reduce code duplication:

```cpp
#define MLLM_CPU_KERNEL_TEST_GEN_TESTS(OpName, EnumDType, CDType, __Op) \
  bool OpName(const std::vector<mllm::Tensor::shape_t>& shapes) { \
    /* Implementation */ \
  }

// Usage
MLLM_CPU_KERNEL_TEST_GEN_TESTS(AddFloat32Test, kFloat32, mllm_fp32_t, +)
```

## Building and Running Tests

Follow the build instructions in [build_mllm.md](./build_mllm.md) to compile the project with tests:

```bash
# For macOS (Apple Silicon)
python task.py tasks/build_osx_apple_silicon.yaml

# For Linux (x86)
python task.py tasks/build_x86.yaml
```

After building, run the tests:

```bash
cd build
ctest
```

Or run specific test executables:

```bash
./tests/cpu/KernelTest
```

## Best Practices

1. **Consistent Naming**: Use consistent naming conventions for test functions and classes
2. **Comprehensive Coverage**: Test all supported data types and common tensor shapes
3. **Clear Reference Implementation**: Make sure the reference implementation is obviously correct
4. **Informative Error Messages**: Provide clear output when tests fail
5. **Modular Design**: Separate test logic into reusable components
6. **Performance Considerations**: Keep test execution time reasonable

## Common Patterns

### Element-wise Operations

For element-wise operations, follow the pattern in `ElementwiseKernelTest`:
- Generate random input tensors
- Compute reference result with a simple loop
- Compare with kernel output using `allClose`

### Reduction Operations

For reduction operations:
- Create input tensors with known values
- Compute expected result manually
- Verify the reduced dimensions match expectations

### Shape Transformation Operations

For operations that change tensor shapes:
- Verify output shapes are correct
- Check that data is transformed as expected
- Ensure memory layout is handled properly

This guide should help AI agents generate appropriate CPU kernel tests for any user-specified kernel in the mllm project.