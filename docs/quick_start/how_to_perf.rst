How to perf modules
====================

MLLM provides built-in performance profiling capabilities based on Perfetto, 
which allows you to analyze the execution performance of your models and modules.

Prerequisites
-------------

To use the performance profiling feature, MLLM must be built with Perfetto support.
This is typically enabled by default in builds that include performance analysis capabilities.

Basic Usage
-----------

To profile your MLLM application, you need to add a few calls to your code:

1. Start profiling at the beginning of your main function or where you want to start measuring
2. Stop profiling at the end of the section you want to measure
3. Save the profiling results to a file

Example
-------

Here's a simple example of how to use performance profiling in your MLLM application:

.. code-block:: cpp

    #include "mllm/mllm.hpp"
    
    int main() {
        mllm::initializeContext();
        
        // Start performance profiling
        #ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::start();
        #endif
        
        // Your model code here
        // ...
        
        // Stop performance profiling
        #ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::stop();
        mllm::perf::saveReport("perf_trace.perfetto");
        #endif
        
        mllm::shutdownContext();
        return 0;
    }

In a more complete example, similar to the Qwen2 VL example:

.. code-block:: cpp

    #include "mllm/mllm.hpp"
    
    int main(int argc, char** argv) {
        mllm::initializeContext();
        
        #ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::start();
        #endif
        
        // Load and run your model
        // ...
        
        #ifdef MLLM_PERFETTO_ENABLE
        mllm::perf::stop();
        mllm::perf::saveReport("model_perf.perfetto");
        #endif
        
        mllm::shutdownContext();
        return 0;
    }

Analyzing Results
-----------------

After running your application, you'll get a `.perfetto` file that can be opened 
with the Perfetto UI at https://ui.perfetto.dev/. This interface allows you to:

- View timeline of operations
- Analyze execution time of different components
- Identify performance bottlenecks
- Examine memory usage patterns

Performance Categories
----------------------

MLLM's performance tracing is organized into several categories:

- ``mllm.func_lifecycle``: Function lifecycle events
- ``mllm.tensor_lifecycle``: Tensor creation, allocation and destruction
- ``mllm.kernel``: Computational kernel execution
- ``mllm.ar_step``: Auto-regressive steps in language models

These categories help you filter and analyze specific aspects of your model's performance.

Best Practices
--------------

1. Only enable profiling when needed, as it may impact performance
2. Use descriptive names for your trace files
3. Profile representative workloads to get meaningful results
4. Remember to call both ``perf::start()`` and ``perf::stop()`` to ensure proper tracing
