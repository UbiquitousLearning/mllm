How to Add a New Operator in MLLM
=================================

This guide will walk you through the process of adding a new operator to the MLLM framework. 
We'll cover all the necessary steps from defining the operator type to implementing it for different backends.

Overview
--------

Adding a new operator to MLLM involves several steps:

1. Define the operator type in ``OpTypes.hpp``
2. Create the operator interface in ``mllm/core/aops/``
3. Implement the operator for each backend in ``mllm/backends/*/ops/``
4. Register the operator factory in the backend
5. Add the operator to IR (Intermediate Representation) if needed

Step 1: Define the Operator Type
--------------------------------

First, you need to add your operator type to the ``OpTypes`` enum in ``mllm/core/OpTypes.hpp``:

.. code-block:: cpp

    enum class OpTypes : int32_t {
      kOpType_Start = 0,
      // ... existing operators ...
      kMyCustomOp,      // Add your operator here
      // ... other operators ...
      kOpType_End,
    };

Also add it to the ``optype2Str`` function:

.. code-block:: cpp

    inline std::string optype2Str(OpTypes type) {
      switch (type) {
        // ... existing cases ...
        case OpTypes::kMyCustomOp: return "MyCustomOp";
        // ... other cases ...
        default: return "Unknown";
      }
    }

Step 2: Create the Operator Interface
-------------------------------------

Create a new header file in ``mllm/core/aops/`` for your operator. For example, ``MyCustomOp.hpp``:

.. code-block:: cpp

    // Copyright (c) MLLM Team.
    // Licensed under the MIT License.

    #pragma once

    #include "mllm/core/BaseOp.hpp"
    #include "mllm/core/ParameterFile.hpp"

    namespace mllm::aops {

    struct MyCustomOpOptions : public BaseOpOptions<MyCustomOpOptions> {
      // Add any options/parameters your operator needs
      int param1 = 0;
      float param2 = 1.0f;
    };

    class MyCustomOp : public BaseOp {
     public:
      explicit MyCustomOp(const MyCustomOpOptions& options);

      void load(const ParameterFile::ptr_t& ploader) override;

      void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

      void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

      void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

      void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

     protected:
      MyCustomOpOptions options_;
    };

    }  // namespace mllm::aops

Then create the implementation file ``MyCustomOp.cpp``:

.. code-block:: cpp

    // Copyright (c) MLLM Team.
    // Licensed under the MIT License.

    #include "mllm/core/aops/MyCustomOp.hpp"
    #include "mllm/core/BaseOp.hpp"
    #include "mllm/core/Tensor.hpp"
    #include "mllm/utils/Common.hpp"
    #include "mllm/compile/ir/linalg/Op.hpp"

    namespace mllm::aops {

    MyCustomOp::MyCustomOp(const MyCustomOpOptions& options) : BaseOp(OpTypes::kMyCustomOp), options_(options) {}

    void MyCustomOp::load(const ParameterFile::ptr_t& ploader) { 
      // Load parameters if needed
      MLLM_EMPTY_SCOPE; 
    }

    void MyCustomOp::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
      auto ir_ctx = (ir::IRContext*)trace_context;
      auto i_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, inputs);
      auto o_irs = ir::tensor::wrapTensors2TensorIR(ir_ctx, outputs);
      ir_ctx->create<ir::linalg::MyCustomOp>(shared_from_this(), i_irs, o_irs);
    }

    void MyCustomOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
      NYI("MyCustomOp::forward not implemented in aops base.");
    }

    void MyCustomOp::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
      // Define output tensor shapes based on input shapes
      // Example for an operation that preserves shape:
      outputs.emplace_back(Tensor::empty(inputs[0].shape(), inputs[0].dtype(), inputs[0].device()));
    }

    void MyCustomOp::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
      BaseOp::setup(inputs, outputs);
    }

    }  // namespace mllm::aops

Step 3: Implement Backend Support
---------------------------------

For each backend you want to support, create implementation files in ``mllm/backends/*/ops/``.

For CPU backend, create ``mllm/backends/cpu/ops/MyCustomOp.hpp``:

.. code-block:: cpp

    // Copyright (c) MLLM Team.
    // Licensed under the MIT License.

    #pragma once

    #include "mllm/core/BaseOp.hpp"
    #include "mllm/core/aops/MyCustomOp.hpp"

    namespace mllm::cpu {

    class CPUMyCustomOp final : public aops::MyCustomOp {
     public:
      explicit CPUMyCustomOp(const aops::MyCustomOpOptions& options);

      void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    };

    class CPUMyCustomOpFactory : public TypedOpFactory<OpTypes::kMyCustomOp, aops::MyCustomOpOptions> {
     public:
      std::shared_ptr<BaseOp> createOpImpl(const aops::MyCustomOpOptions& options) override {
        return std::make_shared<CPUMyCustomOp>(options);
      }
    };

    }  // namespace mllm::cpu

And the implementation ``mllm/backends/cpu/ops/MyCustomOp.cpp``:

.. code-block:: cpp

    // Copyright (c) MLLM Team.
    // Licensed under the MIT License.

    #include "mllm/backends/cpu/ops/MyCustomOp.hpp"

    namespace mllm::cpu {

    CPUMyCustomOp::CPUMyCustomOp(const aops::MyCustomOpOptions& options) : aops::MyCustomOp(options) {}

    void CPUMyCustomOp::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
      auto& input = inputs[0];
      auto& output = outputs[0];

      // Implement your operator logic here
      // Example implementation (element-wise operation):
      auto dtype = input.dtype();
      switch (dtype) {
        case kFloat32: {
          auto input_ptr = input.ptr<float>();
          auto output_ptr = output.ptr<float>();
          for (int i = 0; i < input.numel(); ++i) {
            // Your custom operation
            output_ptr[i] = input_ptr[i] * options_.param2 + options_.param1;
          }
          break;
        }
        // Add cases for other data types as needed
        default:
          NYI("MyCustomOp not supported for data type: {}", nameOfType(dtype));
      }
    }

    }  // namespace mllm::cpu

Step 4: Register the Operator Factory
-------------------------------------

Add your operator factory to the backend registration. For CPU backend, this is typically done in the backend initialization code:

.. code-block:: cpp

    // In your backend initialization code
    backend->regOpFactory<CPUMyCustomOpFactory>();

Step 5: Add to IR (Intermediate Representation)
---------------------------------------------------

If you need to support graph tracing and compilation, add your operator to the IR system:

1. Add your operator to ``mllm/compile/ir/linalg/Op.hpp``:

.. code-block:: cpp

    // In the LINALG_AOPS_DEFINE section
    LINALG_AOPS_DEFINE(MyCustomOp, MYCUSTOMOP);

2. Make sure to include your new operator header where appropriate.

Usage Example
-------------

After implementing your operator, you can use it like this:

.. code-block:: cpp

    #include "mllm/core/aops/MyCustomOp.hpp"

    // Create options
    auto options = mllm::aops::MyCustomOpOptions{};
    options.param1 = 10;
    options.param2 = 2.0f;

    // Create tensors
    auto input = mllm::Tensor::random({1, 3, 224, 224}, -1.0, 1.0, mllm::kFloat32, mllm::kCPU);

    // Execute operator
    auto output = mllm::Context::instance().buildOpAndSubmitTask(
        mllm::OpTypes::kMyCustomOp,
        options,
        {input}
    );

Best Practices
--------------

1. **Follow naming conventions**: Use the established naming patterns in the codebase
2. **Handle all data types**: Ensure your operator works with all relevant data types
3. **Memory management**: Properly handle tensor allocation and deallocation
4. **Error handling**: Implement appropriate error checking and handling
5. **Documentation**: Comment your code clearly
6. **Testing**: Write tests for your new operator in the ``tests/`` directory

Conclusion
----------

Adding a new operator to MLLM requires implementing the interface, backend-specific logic, and proper registration. 
Follow the patterns established by existing operators, and make sure to test your implementation thoroughly.
