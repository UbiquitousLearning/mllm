How to Add New Data Types
=========================

This guide explains how to add new data types to the MLLM framework, including both regular data types and quantized data types.

Overview
--------

MLLM supports various data types including regular floating-point and integer types, as well as quantized types for model compression. All supported data types are defined in the ``DataTypes`` enum in ``mllm/core/DataTypes.hpp``.

The process of adding a new data type involves:

1. Defining the data type in the enum
2. Implementing the data type structure (for quantized types)
3. Adding type information
4. Registering the type in utility functions
5. Implementing quantization/dequantization functions (for quantized types)

Adding a Regular Data Type
--------------------------

1. Add the new data type to the ``DataTypes`` enum:

   In ``mllm/core/DataTypes.hpp``, add your new type to the enum:

   .. code-block:: cpp

      enum DataTypes : int32_t {
        // ... existing types ...
        kMyNewType = 136,  // Use the next available ID
      };

2. Add type information:

   Still in ``mllm/core/DataTypes.hpp``, add the type information specialization:

   .. code-block:: cpp

      MLLM_DEFINE_BASIC_TYPE_INFO(my_new_type_t, zero_val, one_val, max_val, min_val, "MyNewType");

3. Register in utility functions:

   In ``mllm/core/DataTypes.cpp``, add your type to the switch statements in functions like ``lanesOfType``, `[bytesOfType](file:///Volumes/D/mllm/mllm/core/DataTypes.hpp#L547-L547)`, and ``nameOfType``:

   .. code-block:: cpp

      size_t lanesOfType(DataTypes dtype) {
          // ...
          CASE(kMyNewType)
          // ...
      }

Adding a Quantized Data Type
----------------------------

Adding a quantized data type is more involved as it requires defining the data structure and implementing quantization/dequantization functions.

1. Define the quantized data structure:

   In ``mllm/core/DataTypes.hpp``, define your quantized block structure:

   .. code-block:: cpp

      using block_my_q_t = struct {
        mllm_fp16_t d;         // Scaling factor
        uint8_t qs[32];        // Quantized values
      };
      using mllm_block_my_q_t = block_my_q_t;
      static_assert(sizeof(block_my_q_t) == sizeof(mllm_fp16_t) + 32, "wrong my_q block size/padding");

2. Add the data type to the enum:

   .. code-block:: cpp

      enum DataTypes : int32_t {
        // ... existing types ...
        kMyQuantizedType = 136,
      };

3. Add type information for both the block and enum:

   .. code-block:: cpp

      MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_my_q_t, 32, "MyQuantizedType");
      MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kMyQuantizedType, mllm_block_my_q_t);

4. Register in utility functions:

   In ``mllm/core/DataTypes.cpp``, add your type to the switch statements:

   .. code-block:: cpp

      size_t lanesOfType(DataTypes dtype) {
          // ...
          CASE(kMyQuantizedType)
          // ...
      }

5. Implement quantization functions:

   Create new files in ``mllm/backends/cpu/kernels/common/quantize/``:

   In ``quantize_my_q.cpp``:

   .. code-block:: cpp

      #include "mllm/core/DataTypes.hpp"
      #include <cmath>

      void quantize_row_my_q(const float* x, void* vy, int k) {
          // Implementation of quantization
      }

      void dequantize_row_my_q(const void* vx, float* y, int k) {
          // Implementation of dequantization
      }

6. Register quantization functions:

   In the appropriate CPU backend files, add function pointers to your quantization functions.

7. Add to tensor operations:

   Update tensor operations to support your new data type, particularly in files like ``CastTypeOp.cpp``.

Adding Support in Quantizer Tool
--------------------------------

To make your new data type available in the quantizer tool:

1. Update the quantizer schema files in ``tools/mllm-quantizer/schema/``.
2. Add pattern matching rules in the quantizer configuration files.

Testing Your New Data Type
--------------------------

After implementing your new data type:

1. Create unit tests in the ``tests/`` directory.
2. Test tensor creation and basic operations.
3. Test quantization and dequantization if applicable.
4. Test model loading and saving with your new type.

Best Practices
--------------

1. Always use the next available ID in the ``DataTypes`` enum to maintain compatibility.
2. Use appropriate static assertions to ensure correct structure sizes.
3. Follow existing naming conventions.
4. Implement both quantization and dequantization functions for quantized types.
5. Add comprehensive tests for your new data type.
6. Document your new data type in the code and update this guide if necessary.

Example
-------

Here's a complete example of adding a simple 4-bit quantized type:

.. code-block:: cpp

   // In DataTypes.hpp
   #define QK_MY 32
   
   using block_my_4bit = struct {
     mllm_fp16_t d;
     uint8_t qs[QK_MY / 2];
   };
   using mllm_block_my_4bit_t = block_my_4bit;
   
   enum DataTypes : int32_t {
     // ... existing types ...
     kMy4Bit = 136,
   };
   
   MLLM_DEFINE_QUANT_TYPE_INFO(mllm_block_my_4bit_t, QK_MY, "My4Bit");
   MLLM_DEFINE_SELF_TYPE_INFO(DataTypes::kMy4Bit, mllm_block_my_4bit_t);

This guide provides the essential steps to add new data types to MLLM. For specific implementation details, refer to existing data types in the codebase.
