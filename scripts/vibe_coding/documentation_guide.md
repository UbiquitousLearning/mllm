# Documentation Update Guide for MLLM Project

This guide provides instructions for AI agents on how to update documentation when modifying files in the MLLM project. Proper documentation is essential for maintaining code quality and helping developers understand the system.

## Core Documentation Principles

1. **Language**: All documentation must be written in English
2. **Format**: Use reStructuredText (.rst) format for API documentation
3. **Location**: Documentation files are located in the `docs/api/` directory
4. **Style**: Follow the existing documentation style and structure

## When to Update Documentation

Documentation updates are required when making changes to the following directories:

### 1. Core API Changes (`/mllm/core/`)

When modifying files in the core module directory:
- **Reference files**: Look at existing API docs in `docs/api/tensor.rst`
- **Format**: reStructuredText (.rst)
- **Location**: Update or create corresponding files in `docs/api/`
- **Content**: Document all public classes, methods, and functions with:
  - Function signatures
  - Parameter descriptions
  - Return value descriptions
  - Usage examples (if applicable)

Example:
```rst
.. cpp:function:: Tensor Tensor::zeros(const std::vector<int32_t>& shape, DataTypes dtype = kFloat32, DeviceTypes device = kCPU)

   Creates a tensor filled with zeros.

   :param shape: Dimensions of the tensor
   :param dtype: Data type (default: kFloat32)
   :param device: Target device (default: kCPU)
   :return: New tensor with initialized zero values
```

### 2. Neural Network Layers (`/mllm/nn/layers/`)

When adding or modifying neural network layers:
- **Reference files**: See `docs/api/nn.rst`
- **Format**: reStructuredText (.rst)
- **Location**: Update `docs/api/nn.rst`
- **Content**: Document each layer with:
  - Class description
  - Constructor parameters
  - Public methods
  - Example usage

### 3. Functional API (`/mllm/nn/Functional.*`)

When modifying the functional API:
- **Reference files**: See `docs/api/functional.rst`
- **Format**: reStructuredText (.rst)
- **Location**: Update `docs/api/functional.rst`
- **Content**: Document each function with:
  - Function purpose
  - Parameter descriptions
  - Return value description

### 4. Models (`/mllm/models/`)

When adding or modifying models:
- **Reference files**: See `docs/api/argeneration.rst`
- **Format**: reStructuredText (.rst)
- **Location**: Update or create model-specific documentation in `docs/api/`
- **Content**: Document:
  - Model class and inheritance
  - Key methods and their parameters
  - Model-specific configurations

### 5. Main Headers (`/mllm/mllm.*`)

When modifying the main MLLM headers:
- **Reference files**: See `docs/api/mllm.rst`
- **Format**: reStructuredText (.rst)
- **Location**: Update `docs/api/mllm.rst`
- **Content**: Document:
  - Global functions
  - Macros
  - Utility functions

## Documentation Structure

All API documentation follows this structure:

1. **Page Title**: Use `===` underline
2. **Section Titles**: Use `---` underline
3. **Subsection Titles**: Use `^^^^` underline
4. **Class Documentation**:
   ```rst
   .. cpp:class:: ClassName
   
      Brief description of the class
   ```
5. **Function Documentation**:
   ```rst
   .. cpp:function:: return_type function_name(parameters)
   
      Brief description of what the function does
      
      :param parameter_name: Description of parameter
      :return: Description of return value
   ```

## How to Add New Documentation

1. **Identify the component** you're documenting
2. **Check existing documentation** in `docs/api/` for similar components
3. **Follow the established pattern** from existing files
4. **Add your component** to the appropriate .rst file or create a new one
5. **Update the index** in `docs/api/index.rst` to include your new file
6. **Verify the structure** matches existing documentation

## Special Considerations

### Cross-references

Use cross-references to link related components:
```rst
:cpp:class:`Tensor` 
:cpp:func:`Tensor::zeros`
```

### Code Examples

Include brief code examples when helpful:
```rst
.. code-block:: cpp

   #include "mllm/core/Tensor.hpp"
   
   auto tensor = Tensor::zeros({10, 10});
```

### Template Functions

Document template functions with their template parameters:
```rst
.. cpp:function:: template<int32_t RET_NUM> std::array<Tensor, RET_NUM> split(const Tensor& x, int32_t split_size_or_sections, int32_t dim)

   Split a tensor into a fixed number of chunks with same size along a given dimension.
```

## Validation

Before submitting documentation changes:

1. Ensure all public APIs are documented
2. Verify parameter and return types are correctly specified
3. Check that cross-references are valid
4. Confirm the documentation builds without errors
5. Make sure examples are accurate and complete

## Common Documentation Patterns

### Class Documentation Template
```rst
ClassName
---------

.. cpp:class:: ClassName

   Brief description of the class purpose and usage.

   .. cpp:function:: ClassName::ClassName(parameters)

      Constructor description.

      :param parameter: Parameter description

   .. cpp:function:: return_type ClassName::method_name(parameters)

      Method description.

      :param parameter: Parameter description
      :return: Return value description
```

### Function Documentation Template
```rst
.. cpp:function:: return_type function_name(parameters)

   Brief description of what the function does.

   :param param1: Description of first parameter
   :param param2: Description of second parameter with default value (default: value)
   :return: Description of return value
```

Following this guide ensures consistent, high-quality documentation that helps developers effectively use and contribute to the MLLM project.