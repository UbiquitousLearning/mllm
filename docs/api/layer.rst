Layer API
=========

The Layer class represents a basic computational unit in MLLM's neural network framework. Layers are typically used to implement specific operations like linear transformations, convolutions, or activation functions.

.. code-block:: cpp

   #include "mllm/nn/Layer.hpp"

Base Class
----------

.. cpp:class:: Layer

   Base class for neural network layers. Layers are typically used to implement specific operations.

Constructors
------------

.. cpp:function:: explicit Layer::Layer(const LayerImpl::ptr_t& impl)

   Constructor with LayerImpl pointer.

   :param impl: Shared pointer to LayerImpl instance

.. cpp:function:: template<typename T> Layer::Layer(OpTypes op_type, const T& cargo)

   Constructor with operation type and options.

   :param op_type: Operation type for the layer
   :param cargo: Options for the operation

Core Methods
------------

.. cpp:function:: LayerImpl::ptr_t Layer::impl() const

   Get the underlying LayerImpl pointer.

   :return: Shared pointer to LayerImpl

.. cpp:function:: std::vector<Tensor> Layer::__main(const std::vector<Tensor>& inputs)

   Main execution method for the layer.

   :param inputs: Input tensors
   :return: Output tensors

.. cpp:function:: OpTypes Layer::opType() const

   Get the operation type of the layer.

   :return: Operation type

.. cpp:function:: BaseOpOptionsBase& Layer::refOptions()

   Get reference to the layer's options.

   :return: Reference to BaseOpOptionsBase

.. cpp:function:: Layer& Layer::to(DeviceTypes device_type)

   Move the layer to specified device.

   :param device_type: Target device type (kCPU, kCUDA, etc.)
   :return: Reference to this layer

.. cpp:function:: void Layer::__fmt_print(std::stringstream& ss)

   Format print information about the layer.

   :param ss: String stream to write formatted output

Helper Macros
-------------

.. c:macro:: MLLM_LAYER_ANY_INPUTS_1_OUTPUTS_FORWARD

   Macro for defining operator() with any number of inputs and 1 output.

.. c:macro:: MLLM_LAYER_ANY_INPUTS_2_OUTPUTS_FORWARD

   Macro for defining operator() with any number of inputs and 2 outputs.

.. c:macro:: MLLM_LAYER_ANY_INPUTS_3_OUTPUTS_FORWARD

   Macro for defining operator() with any number of inputs and 3 outputs.