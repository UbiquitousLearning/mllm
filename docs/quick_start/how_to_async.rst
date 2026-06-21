How to run modules async
=========================

MLLM provides asynchronous execution capabilities that allow you to run multiple modules or operations concurrently, which can significantly improve performance in certain scenarios.

Basic Usage
-----------

To run modules asynchronously, you can use the ``mllm::async::fork`` function to create asynchronous tasks, and then use ``mllm::async::wait`` to wait for their completion.

Here's a basic example:

.. code-block:: cpp

   #include "mllm/mllm.hpp"
   
   // Initialize the context first
   mllm::initializeContext();
   
   // Create your module (neural network)
   auto net = FooNet("foo_net");
   // ... load parameters ...
   
   // Fork asynchronous tasks
   auto future_0 = mllm::async::fork(net, Tensor::empty({1, 12, 1024, 1024}, kFloat32).alloc());
   auto future_1 = mllm::async::fork(net, Tensor::empty({1, 12, 1024, 1024}, kFloat32).alloc());
   
   // Wait for completion and get results
   auto [outs_0, outs_1] = mllm::async::wait(future_0, future_1);

The ``fork`` function
-----------------------

The ``mllm::async::fork`` function creates an asynchronous task without immediately executing it:

.. code-block:: cpp

   template<typename __Module, typename... __Args>
   std::pair<TaskResult::sender_t, Task::ptr_t> fork(__Module& module, __Args&&... args)

It takes a module and its input arguments, and returns a pair containing a sender (for synchronization) and a task pointer (for accessing results).

The ``wait`` function
-----------------------

The ``mllm::async::wait`` function is used to wait for the completion of asynchronous tasks:

.. code-block:: cpp

   // For a single task
   std::vector<Tensor> wait(std::pair<TaskResult::sender_t, Task::ptr_t>& sender);
   
   // For multiple tasks
   template<typename... __Args>
   std::array<std::vector<Tensor>, sizeof...(__Args)> wait(__Args&&... args)

It blocks until the specified tasks are completed and returns their results.

Notification
--------------------

Please note that nesting async execution within a forked module may cause deadlocks. This is because each fork occupies a thread in the thread pool, and excessive nesting can consume a large number of thread resources.

Remember to always call ``mllm::initializeContext()`` before using asynchronous features, and properly manage the lifetime of your modules and tensors.


Complete Example
----------------

Here's a complete example showing how to use asynchronous execution:

.. code-block:: cpp

   #include "mllm/mllm.hpp"
   
   using namespace mllm;
   
   class FooNet final : public nn::Module {
     nn::Linear linear_0;
     nn::Linear linear_1;
     nn::Linear linear_2;
     nn::Linear linear_3;
   
    public:
     explicit FooNet(const std::string& name) : nn::Module(name) {
       linear_0 = reg<nn::Linear>("linear_0", /*in_channels*/ 1024, /*out_channels*/ 2048);
       linear_1 = reg<nn::Linear>("linear_1", /*in_channels*/ 1024, /*out_channels*/ 2048);
       linear_2 = reg<nn::Linear>("linear_2", /*in_channels*/ 1024, /*out_channels*/ 2048);
       linear_3 = reg<nn::Linear>("linear_3", /*in_channels*/ 1024, /*out_channels*/ 2048);
     }
   
     std::vector<Tensor> forward(const std::vector<Tensor>& inputs) override {
       return {
           linear_0(inputs[0]),
           linear_1(inputs[0]),
           linear_2(inputs[0]),
           linear_3(inputs[0]),
       };
     }
   };
   
   int main() {
     mllm::initializeContext();
     {
       auto net = FooNet("foo_net");
   
       // Make some fake weights
       auto params = ParameterFile::create();
       for (int i = 0; i < 4; ++i) {
         auto name = "foo_net.linear_" + std::to_string(i);
         auto w = Tensor::empty({2048, 1024}).setMemType(kParamsNormal).setName(name + ".weight").alloc();
         auto b = Tensor::empty({2048}).setMemType(kParamsNormal).setName(name + ".bias").alloc();
         params->push(w.name(), w);
         params->push(b.name(), b);
       }
       net.load(params);
   
       // Async run.
       // The net will not run, until mllm::async::wait is called.
       auto future_0 = mllm::async::fork(net, Tensor::empty({1, 12, 1024, 1024}, kFloat32).alloc());
       auto future_1 = mllm::async::fork(net, Tensor::empty({1, 12, 1024, 1024}, kFloat32).alloc());
   
       // Run future_0 and future_1 async.
       auto [outs_0, outs_1] = mllm::async::wait(future_0, future_1);
   
       mllm::print(outs_0[0].shape(), outs_0[1].shape(), outs_0[2].shape(), outs_0[3].shape());
       mllm::print(outs_1[0].shape(), outs_1[1].shape(), outs_1[2].shape(), outs_1[3].shape());
     }
     mllm::memoryReport();
   }
