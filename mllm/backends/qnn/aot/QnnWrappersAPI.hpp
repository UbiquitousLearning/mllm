// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <dlfcn.h>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>

#include <QnnTypes.h>
#include <QnnCommon.h>
#include <QnnContext.h>
#include <QnnInterface.h>
#include <QnnSdkBuildId.h>
#include <HTP/QnnHtpDevice.h>
#include <System/QnnSystemInterface.h>

#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/backends/qnn/QNNModel.hpp"
#include "mllm/backends/qnn/QNNUtils.hpp"

namespace mllm::qnn::aot {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp, va_list argp);

// Collection of symbols that we need to load from qnn dyn lib.
struct QnnFuncSymbols {
  using QnnInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnInterface_t*** providerList, uint32_t* numProviders);
  using QnnSystemInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnSystemInterface_t*** providerList,
                                                                   uint32_t* numProviders);

  QNN_INTERFACE_VER_TYPE qnn_interface_;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
};

class QnnAOTNodeTensor : public std::enable_shared_from_this<QnnAOTNodeTensor> {
 public:
  using ptr_t = std::shared_ptr<QnnAOTNodeTensor>;

  static inline ptr_t create(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight = false) {
    return std::make_shared<QnnAOTNodeTensor>(v, force_static_weight);
  }

  explicit QnnAOTNodeTensor(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight = false);

  std::shared_ptr<mllm::qnn::QNNTensorWrapper> getWrapper() { return tensor_wrapper_; }

 private:
  Qnn_TensorType_t parseQnnTensorTypeFromIR(const ir::tensor::TensorValue::ptr_t& v);

  Qnn_DataType_t parseQnnDataTypeFromIR(const ir::tensor::TensorValue::ptr_t& v);

  std::string parseQnnTensorNameFromIR(const ir::tensor::TensorValue::ptr_t& v);

  Qnn_QuantizeParams_t parseQnnQuantizeParamFromIR(const ir::tensor::TensorValue::ptr_t& v);

  // intend for per-channel and LPBQ quantization
  void setupComplexTensorQuantization(const ir::tensor::TensorValue::ptr_t& v);

  std::shared_ptr<mllm::qnn::QNNTensorWrapper> tensor_wrapper_;
};

class QnnAOTNodeOperation : public std::enable_shared_from_this<QnnAOTNodeOperation> {
 public:
  using ptr_t = std::shared_ptr<QnnAOTNodeOperation>;

  static inline ptr_t create(const std::string& op_name) {
    auto ret = std::make_shared<QnnAOTNodeOperation>();
    ret->op_name_ = op_name;
    return ret;
  }

  QnnAOTNodeOperation::ptr_t addInputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ins);

  QnnAOTNodeOperation::ptr_t addOutputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ous);

  QnnAOTNodeOperation::ptr_t emplaceInput(const QnnAOTNodeTensor::ptr_t& input);

  QnnAOTNodeOperation::ptr_t emplaceOutput(const QnnAOTNodeTensor::ptr_t& output);

  QnnAOTNodeOperation::ptr_t addParamScalar(const std::vector<std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>>& params);

  QnnAOTNodeOperation::ptr_t emplaceParamScalar(const std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>& param);

  QnnAOTNodeOperation::ptr_t addParamTensor(const std::vector<std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>>& params);

  QnnAOTNodeOperation::ptr_t emplaceParamTensor(const std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>& param);

  QnnAOTNodeOperation::ptr_t setOpName(const std::string& op_name);

  QnnAOTNodeOperation::ptr_t setName(const std::string& name);

  std::string getName();

  QnnAOTNodeOperation::ptr_t setPackageName(const std::string& package_name);

  std::string name_;
  std::string op_name_;
  std::string package_name_ = "qti.aisw";
  std::vector<std::shared_ptr<mllm::qnn::QNNParamScalarWrapper>> param_scalar;
  std::vector<std::shared_ptr<mllm::qnn::QNNParamTensorWrapper>> param_tensor;
  std::vector<QnnAOTNodeTensor::ptr_t> inputs;
  std::vector<QnnAOTNodeTensor::ptr_t> outputs;
};

struct QnnDeviceAndContext;
class QnnAOTGraph : public std::enable_shared_from_this<QnnAOTGraph> {
 public:
  using ptr_t = std::shared_ptr<QnnAOTGraph>;

  QnnAOTGraph(QNN_INTERFACE_VER_TYPE& qnnInterface, Qnn_BackendHandle_t backendHandle, Qnn_ContextHandle_t contextHandle,
              const std::string& graphName);

  void addOperation(const QnnAOTNodeOperation::ptr_t& qnn_op);

  bool compile();

  bool is_compiled_ = false;
  std::unordered_map<std::string, QnnAOTNodeOperation::ptr_t> op_node_;
  std::unordered_map<std::string, QnnAOTNodeTensor::ptr_t> all_tensors_;

 private:
  std::shared_ptr<mllm::qnn::QNNModel> qnn_model_;
};

struct QnnDeviceAndContext {
  using ptr_t = std::shared_ptr<QnnDeviceAndContext>;

  std::string name_;
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t bk_handle_ = nullptr;
  Qnn_DeviceHandle_t device_handle_ = nullptr;
  QnnBackend_Config_t** bk_cfg_ = nullptr;
  QnnContext_Config_t** qnn_context_config_ = nullptr;
  Qnn_ProfileHandle_t profile_bk_handle_ = nullptr;
  Qnn_ContextHandle_t qnn_ctx_handle_;

  std::unordered_map<std::string, QnnAOTGraph::ptr_t> graphs_;              //< for persistence keep graphs.
  std::unordered_map<std::string, QnnAOTNodeTensor::ptr_t> static_tensor_;  //< for weight sharing.
};

struct QnnDynLibDescriptor {
  std::string lib_name_;
  std::string lib_path_;
  void* handle_ = nullptr;

  template<typename FuncType>
  std::function<FuncType> func(const std::string& symbol_name) {
    if (handle_ == nullptr) { MLLM_ERROR_EXIT(ExitCode::kCoreError, "QnnDynSymbolLoader: handle is nullptr."); }
    auto func_ptr = dlsym(handle_, symbol_name.c_str());
    MLLM_RT_ASSERT(func_ptr != nullptr);
    return (FuncType*)(func_ptr);
  };
};

class QnnDynSymbolLoader {
 public:
  enum DynFlag : int {  // NOLINT performance-enum-size
    kRTLD_NOW = RTLD_NOW,
    kRTLD_LOCAL = RTLD_LOCAL,
    kRTLD_GLOBAL = RTLD_GLOBAL,
  };

  static QnnDynSymbolLoader& instance() {
    static QnnDynSymbolLoader instance;
    return instance;
  }

  ~QnnDynSymbolLoader();

  QnnDynSymbolLoader() = default;

  QnnDynSymbolLoader(const QnnDynSymbolLoader&) = delete;

  QnnDynSymbolLoader& operator=(const QnnDynSymbolLoader&) = delete;

  bool loadQnnDynLib(const std::string& lib_name, int flag);

  bool loadQnnDynLibAtPath(const std::string& path, const std::string& lib_name, int flag);

  inline QnnDynLibDescriptor& operator()(const std::string& lib_name) { return libs_.at(lib_name); }

 private:
  std::unordered_map<std::string, QnnDynLibDescriptor> libs_;
  static const std::vector<std::string> possible_qnn_dyn_lib_paths_;
};

// Device and Dynamic Lib included
class QnnAOTEnv {
 public:
  using ptr_t = std::shared_ptr<QnnAOTEnv>;

  explicit QnnAOTEnv(const QcomTargetMachine& target_machine);

  QnnAOTEnv(const std::string& lib_path, const QcomTargetMachine& target_machine);

  std::shared_ptr<QnnDeviceAndContext> createContext(const std::string& name, bool weights_sharing = false);

  void saveContext(const std::string& name, const std::string& path);

  void destroyContext(const std::string& name);

  // This is for All PUs, such as CPU, GPU, NPU
  std::vector<QnnDevice_PlatformInfo_t*> createDevicePlatformInfo();

  // This function is for NPU only.
  std::vector<QnnDevice_CustomConfig_t> createDecideCustomConfigInfo();

  std::vector<QnnContext_CustomConfig_t> createContextCustomConfig(bool weights_sharing);

  // Functions for build qnn graphs
  QnnAOTGraph::ptr_t captureAOTGraph(const std::string& qnn_context_name, const std::string& g_name);

  void captureAOTNodeOp(const std::string& qnn_context_name, const std::string& graph_name,
                        const QnnAOTNodeOperation::ptr_t& op);

  QnnAOTNodeTensor::ptr_t captureQnnAOTNodeTensor(const std::string& qnn_context_name, const std::string& graph_name,
                                                  const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight = false);

  inline QnnFuncSymbols& getFuncSymbol() { return qnn_htp_func_symbols_; }

  std::shared_ptr<QnnDeviceAndContext> getContext(const std::string& name);

 private:
  void _setup(const std::string& path = "");

  QcomTargetMachine target_machine_;
  QnnFuncSymbols qnn_htp_func_symbols_;
  std::unordered_map<std::string, std::shared_ptr<QnnDeviceAndContext>> contexts_;

  // device config for all to use
  std::vector<QnnDevice_Config_t> target_machine_qnn_config_;
  std::vector<const QnnDevice_Config_t*> target_machine_qnn_config_ptrs_;

  // void* handle that should be freed when QnnAOTEnv end
  std::vector<void*> unreachable_handle_;
};

}  // namespace mllm::qnn::aot
