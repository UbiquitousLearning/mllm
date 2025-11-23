Op Plugin System
=================

mllm's operator plugin system provides a flexible mechanism for developers to extend the framework with new operators. The system supports two main methods for operator extension:

1.  **In-tree Operator Registration**: Suitable for operators tightly integrated with a specific backend. These operators are typically compiled into the mllm core library along with the backend code.
2.  **Out-of-tree Plugin Operator**: Allows developers to develop operators in separate code repositories and compile them into dynamic-link libraries (e.g., `.so`, `.dylib`, `.dll`). The mllm framework can automatically discover and load these plugins at runtime, enabling dynamic extension of functionality.

In-tree Operator Registration
-----------------------------

For a backend already integrated into the mllm project, you can register backend-specific operators by calling `Context::registerCustomizedOp` during the backend initialization process.

Taking the QNN backend as an example, its operator registration is done in `mllm/backends/qnn/Register.cpp`:

.. code-block:: cpp

    // mllm/backends/qnn/Register.cpp

    namespace mllm {

    void initQnnBackend() {
      mllm_RT_ASSERT(isQnnAvailable());
      auto& ctx = Context::instance();

      // 1. Register backend
      auto backend = std::make_shared<qnn::QNNBackend>();
      ctx.registerBackend(backend);

      // ... other initializations ...

      // 3. Register QNN custom ops
      Context::instance().registerCustomizedOp(kQNN, "DequantizeAdd",
                                               std::shared_ptr<BaseOpFactory>((BaseOpFactory*)(new qnn::DequantizeAddFactory())));
    }
    }  // namespace mllm

As shown above, the `initQnnBackend` function, while initializing the QNN backend, also registers the `DequantizeAdd` operator and its factory, `DequantizeAddFactory`, to the `kQNN` device type via `registerCustomizedOp`. This way, when a `DequantizeAdd` operator appears in the model and is scheduled to run on the QNN backend, the framework can find and create the corresponding operator instance.

Out-of-tree Plugin Operator
---------------------------

The external plugin mechanism is a more powerful and flexible part of the mllm operator system. It allows developers to create independent operator libraries without modifying mllm's main codebase. `mllm-ext-opset` is a typical example that shows how to create a plugin with extra operators for the CPU.

An external operator plugin mainly consists of the following parts:

### 1. Operator Implementation

First, you need to define the specific implementation of the operator. The operator class must inherit from `mllm::plugin::interface::CustomizedOp`.

.. code-block:: cpp

    // mllm-ext-opset/cpu/fa2_swa_sink/FlashAttn2SwaSink.hpp

    namespace mllm::ext_opset::cpu {

    class FlashAttention2SwaSink final : public mllm::plugin::interface::CustomizedOp {
     public:
      explicit FlashAttention2SwaSink(const FlashAttention2SwaSinkOptions& options)
          : CustomizedOp("flash_attention_2_swa_sink"), options_(options) {}

      // ~~~ Must implement the following virtual functions ~~~
      void load(...) override;
      void trace(...) override;
      void forward(...) override;
      void reshape(...) override;
      void setup(...) override;
     protected:
      FlashAttention2SwaSinkOptions options_;
    };

    } // namespace mllm::ext_opset::cpu

### 2. Operator Factory

Each operator needs a corresponding factory class to create operator instances. The factory class must inherit from `mllm::plugin::interface::CustomizedOpFactory`.

.. code-block:: cpp

    // mllm-ext-opset/cpu/fa2_swa_sink/FlashAttn2SwaSink.hpp

    namespace mllm::ext_opset::cpu {

    class FlashAttention2SwaSinkFactory final : public mllm::plugin::interface::CustomizedOpFactory<FlashAttention2SwaSinkOptions> {
     public:
      inline std::shared_ptr<mllm::BaseOp> createOpImpl(const FlashAttention2SwaSinkOptions& cargo) override {
        auto p = std::make_shared<FlashAttention2SwaSink>(cargo);
        p->setOpType(opType());
        return p;
      }
    };

    } // namespace mllm::ext_opset::cpu

### 3. Plugin Descriptor

This is the key to exposing the operator to the mllm framework. Each plugin dynamic library must export a C function named `opPackageDescriptor`. This function returns a pointer to a `PluginOpPackageDescriptor` struct, which contains metadata about the plugin, such as its name, supported device type, a list of included operators, and function pointers to create and free the operator factories.

.. code-block:: cpp

    // mllm-ext-opset/cpu/fa2_swa_sink/FlashAttn2SwaSink.cpp

    mllm_PLUGIN_OP_INTERFACE_DEFINE_BEGIN
    // Function to create the factory
    void* createFlashAttention2SwaSinkFactory() { return new mllm::ext_opset::cpu::FlashAttention2SwaSinkFactory(); };

    // Function to free the factory
    void freeFlashAttention2SwaSinkFactory(void* factory) {
      delete static_cast<mllm::ext_opset::cpu::FlashAttention2SwaSinkFactory*>(factory);
    };

    // Descriptor export function
    void* opPackageDescriptor() {
      auto package = new PluginOpPackageDescriptor{
          .version = mllm_PLUGIN_OP_PACKAGE_DESCRIPTOR_VERSION,
          .name = "mllmExtOpSet.CPU.FlashAttn2SwaSink",
          .device_type = 1, // Corresponds to mllm::kCPU
          .op_factories_count = 1,
          .op_factories_names =
              {
                  "flash_attention_2_swa_sink", // Operator type name
              },
          .op_factory_create_funcs =
              {
                  createFlashAttention2SwaSinkFactory, // Factory creation function pointer
              },
          .op_factory_free_funcs =
              {
                  freeFlashAttention2SwaSinkFactory, // Factory free function pointer
              },
      };
      return package;
    }
    mllm_PLUGIN_OP_INTERFACE_DEFINE_END

### 4. Build Configuration (CMake)

Finally, you need to use CMake to compile the plugin into a `SHARED` library.

.. code-block:: cmake

    # mllm-ext-opset/cpu/fa2_swa_sink/CMakeLists.txt

    add_library(mllmExtOpSet_CPU_FlashAttn2SwaSink SHARED FlashAttn2SwaSink.cpp)
    target_link_libraries(mllmExtOpSet_CPU_FlashAttn2SwaSink PRIVATE mllmRT mllmCPUBackend)
    target_include_directories(mllmExtOpSet_CPU_FlashAttn2SwaSink PRIVATE ${mllm_INCLUDE_DIR})

    install(
      TARGETS mllmExtOpSet_CPU_FlashAttn2SwaSink
      EXPORT mllmTargets
      LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin)

### Plugin Loading Mechanism

When the mllm framework starts, it scans for all dynamic libraries in predefined paths (such as `lib/` or paths specified by environment variables). For each library, it attempts to load it using `dlopen` and looks for the `opPackageDescriptor` symbol using `dlsym`. If the symbol is found, the framework calls it to get the plugin descriptor and registers all operator factories from the plugin into the `Context` based on the information in the descriptor.

This approach decouples external operator plugins from the mllm framework, providing excellent extensibility.