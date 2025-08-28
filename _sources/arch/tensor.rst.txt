Tensor
======

.. figure:: ../_static/img/tensor-storage.png
   :width: 80%
   :alt: Overview
   :align: center

   Figure 1: Tensor Storage and Tensor View.

Memory Layout
-------------

In Mllm, the memory layout of a Tensor is described using the following key concepts:

- **Shape:** Represents the dimensions of the tensor as an array of integers. For example, a tensor with shape ``[3, 4, 5]`` is a three-dimensional tensor with sizes ``3``, ``4``, and ``5`` along its respective dimensions.
- **Stride:** Specifies the number of elements to step in each dimension when moving to the next element in memory. Strides define how to navigate through the memory for each dimension. For instance, a 2D tensor with shape ``[3, 4]`` and strides ``[4, 1]`` means that moving along the first dimension skips ``4`` elements, while moving along the second dimension skips just ``1`` element.
- **Storage Offset:** Indicates the starting offset within the underlying storage where the tensor's data begins. This allows tensors to share the same storage while accessing different sub-regions via distinct offsets.

When calculating a pointer (ptr) to access a specific element in the tensor, we can use the following formula:

.. code-block:: text

   ptr = storage.ptr + (storage_offsets + dot_product(indices, stride)) * sizeof(datatype)