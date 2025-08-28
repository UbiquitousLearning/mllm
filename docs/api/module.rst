Module API
==========

The Module class is a fundamental building block in MLLM's neural network framework. It serves as a container for neural network components and provides functionalities for parameter management, device placement, and execution.

.. code-block:: cpp

   #include "mllm/nn/Module.hpp"

Base Class
----------

.. cpp:class:: Module

   Base class for neural network modules. Modules can contain other modules or layers and manage their parameters and execution.

Constructors
------------

.. cpp:function:: Module::Module()

   Default constructor.

.. cpp:function:: explicit Module::Module(const ModuleImpl::ptr_t& impl)

   Constructor with ModuleImpl pointer.

   :param impl: Shared pointer to ModuleImpl instance

.. cpp:function:: explicit Module::Module(const std::string& name)

   Constructor with module name.

   :param name: Name of the module

Core Methods
------------

.. cpp:function:: ModuleImpl::ptr_t Module::impl() const

   Get the underlying ModuleImpl pointer.

   :return: Shared pointer to ModuleImpl

.. cpp:function:: void Module::to(DeviceTypes device_type)

   Move the module and its parameters to specified device.

   :param device_type: Target device type (kCPU, kCUDA, etc.)

.. cpp:function:: template<typename T, typename... Args> auto Module::reg(const std::string& name, Args&&... args)

   Register a sub-module or layer to this module.

   :param name: Name of the sub-module or layer
   :param args: Arguments for constructing the sub-module or layer
   :return: Registered sub-module or layer

.. cpp:function:: template<typename... Args> std::vector<Tensor> Module::operator()(Args&&... args)

   Execute the module with given inputs.

   :param args: Input tensors and other arguments
   :return: Output tensors

.. cpp:function:: void Module::load(const ParameterFile::ptr_t& param_file)

   Load parameters from a parameter file.

   :param param_file: Shared pointer to ParameterFile

.. cpp:function:: virtual std::vector<Tensor> Module::forward(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args)

   Forward pass of the module. Should be implemented by derived classes.

   :param inputs: Input tensors
   :param args: Additional arguments
   :return: Output tensors

Utility Methods
---------------

.. cpp:function:: void Module::__fmt_print(std::stringstream& ss) const

   Format print information about the module.

   :param ss: String stream to write formatted output

.. cpp:function:: std::vector<Tensor> Module::__main(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args)

   Main execution method that handles preprocessing and postprocessing.

   :param inputs: Input tensors
   :param args: Additional arguments
   :return: Output tensors

.. cpp:function:: void Module::__send_graph_begin(const std::vector<Tensor>& inputs)

   Send graph begin signal for execution tracing.

   :param inputs: Input tensors

.. cpp:function:: void Module::__send_graph_end(const std::vector<Tensor>& inputs)

   Send graph end signal for execution tracing.

   :param inputs: Input tensors

.. cpp:function:: std::vector<Tensor> Module::__trace(const std::vector<Tensor>& inputs, const std::vector<AnyValue>& args)

   Trace execution for compilation or analysis.

   :param inputs: Input tensors
   :param args: Additional arguments
   :return: Output tensors

.. cpp:function:: ParameterFile::ptr_t Module::params(ModelFileVersion v)

   Get parameters of the module.

   :param v: Model file version
   :return: Shared pointer to ParameterFile

.. cpp:function:: void Module::registerBuffer(const std::string& name, const Tensor& tensor)

   Register a buffer tensor that won't be saved with parameters.

   :param name: Name of the buffer
   :param tensor: Buffer tensor

.. cpp:function:: Tensor Module::getBuffer(const std::string& name)

   Get a registered buffer tensor.

   :param name: Name of the buffer
   :return: Buffer tensor

.. cpp:function:: std::string Module::getModuleName() const

   Get the full name of the module.

   :return: Module name