mllm API
========

The mllm.hpp header is the main header file that includes all essential MLLM components. It provides core functionalities for model loading, context management, asynchronous execution, and utility functions.

.. code-block:: cpp

   #include "mllm.hpp"

Core Functions
--------------

.. cpp:function:: void mllm::initializeContext()

   Initialize the MLLM context, register backends, and set up memory management.

.. cpp:function:: void mllm::shutdownContext()

   Shutdown the MLLM context and clean up resources.

.. cpp:function:: void mllm::setRandomSeed(uint64_t seed)

   Set the random seed for reproducible results.

   :param seed: Random seed value

.. cpp:function:: void mllm::setMaximumNumThreads(uint32_t num_threads)

   Set the maximum number of threads for parallel execution.

   :param num_threads: Maximum number of threads

.. cpp:function:: void mllm::setPrintPrecision(int precision)

   Set the floating-point precision for printing tensors.

   :param precision: Number of decimal places

.. cpp:function:: void mllm::setPrintMaxElementsPerDim(int max_elements)

   Set the maximum number of elements to print per dimension.

   :param max_elements: Maximum elements per dimension

.. cpp:function:: void mllm::memoryReport()

   Print a memory usage report.

.. cpp:function:: bool mllm::isOpenCLAvailable()

   Check if OpenCL backend is available.

   :return: True if OpenCL is available, false otherwise

.. cpp:function:: bool mllm::isQnnAvailable()

   Check if QNN backend is available.

   :return: True if QNN is available, false otherwise

.. cpp:function:: void mllm::cleanThisThread()

   Clean up thread-local resources.

.. cpp:function:: SessionTCB::ptr_t mllm::thisThread()

   Get the current thread's session context.

   :return: Shared pointer to SessionTCB

Parameter File Functions
------------------------

.. cpp:function:: ParameterFile::ptr_t mllm::load(const std::string& file_name, ModelFileVersion version = ModelFileVersion::kV1, DeviceTypes map_2_device = kCPU)

   Load a parameter file.

   :param file_name: Path to the parameter file
   :param version: Model file version (default: kV1)
   :param map_2_device: Target device for loading (default: kCPU)
   :return: Shared pointer to ParameterFile

.. cpp:function:: void mllm::save(const std::string& file_name, const ParameterFile::ptr_t& parameter_file, ModelFileVersion version = ModelFileVersion::kV1, DeviceTypes map_2_device = kCPU)

   Save parameters to a file.

   :param file_name: Path to save the parameter file
   :param parameter_file: ParameterFile to save
   :param version: Model file version (default: kV1)
   :param map_2_device: Target device for saving (default: kCPU)

Utility Functions
-----------------

.. cpp:function:: template<typename... Args> void mllm::print(const Args&... args)

   Print arguments to stdout with automatic formatting.

   :param args: Arguments to print

Testing Functions
-----------------

.. cpp:function:: mllm::test::AllCloseResult mllm::test::allClose(const Tensor& a, const Tensor& b, float rtol = 1e-5, float atol = 1e-5, bool equal_nan = false)

   Check if two tensors are close within tolerance.

   :param a: First tensor
   :param b: Second tensor
   :param rtol: Relative tolerance (default: 1e-5)
   :param atol: Absolute tolerance (default: 1e-5)
   :param equal_nan: Whether NaNs should be considered equal (default: false)
   :return: AllCloseResult containing comparison results

.. cpp:class:: mllm::test::AllCloseResult

   Result structure for allClose function.

   .. cpp:member:: bool mllm::test::AllCloseResult::is_close

      True if tensors are close within tolerance

   .. cpp:member:: size_t mllm::test::AllCloseResult::total_elements

      Total number of elements compared

   .. cpp:member:: size_t mllm::test::AllCloseResult::mismatched_elements

      Number of elements that don't match within tolerance

   .. cpp:member:: float mllm::test::AllCloseResult::max_absolute_diff

      Maximum absolute difference

   .. cpp:member:: float mllm::test::AllCloseResult::max_relative_diff

      Maximum relative difference

Async Execution Functions
-------------------------

.. cpp:function:: template<typename __Module, typename... __Args> std::pair<TaskResult::sender_t, Task::ptr_t> mllm::async::fork(__Module& module, __Args&&... args)

   Fork a task for asynchronous execution.

   :param module: Module to execute
   :param args: Arguments for module execution
   :return: Pair of sender and task pointer

.. cpp:function:: std::vector<Tensor> mllm::async::wait(std::pair<TaskResult::sender_t, Task::ptr_t>& sender)

   Wait for a single asynchronous task to complete.

   :param sender: Sender-task pair
   :return: Output tensors

.. cpp:function:: template<typename... __Args> std::array<std::vector<Tensor>, sizeof...(__Args)> mllm::async::wait(__Args&&... args)

   Wait for multiple asynchronous tasks to complete.

   :param args: Sender-task pairs
   :return: Array of output tensors

Signal Handling
---------------

.. cpp:function:: void mllm::__setup_signal_handler()

   Set up signal handlers for graceful shutdown on interruption.

.. cpp:function:: void mllm::__signal_handler(int signal)

   Signal handler function.

   :param signal: Signal number

.. cpp:function:: template<typename Func> int mllm::__mllm_exception_main(Func&& func)

   Exception-safe main function wrapper.

   :param func: User function to execute
   :return: Exit code

.. cpp:function:: const char* mllm::signal_description(int signal)

   Get human-readable description of a signal.

   :param signal: Signal number
   :return: Description string

Macros
------

.. c:macro:: MLLM_MAIN(...)

   Main function macro that sets up signal handlers, initializes context, and provides exception safety.

Performance Functions
---------------------

.. cpp:function:: void mllm::perf::warmup(const ParameterFile::ptr_t& params)

   Warm up the model with given parameters.

   :param params: Parameters for warmup