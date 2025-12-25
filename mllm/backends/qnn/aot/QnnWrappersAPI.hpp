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

#include <QNN/QnnTypes.h>
#include <QNN/QnnCommon.h>
#include <QNN/QnnContext.h>
#include <QNN/QnnInterface.h>
#include <QNN/QnnSdkBuildId.h>
#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/System/QnnSystemInterface.h>

#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/compile/ir/tensor/Value.hpp"
#include "mllm/utils/Common.hpp"

namespace mllm::qnn::aot {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp, va_list argp);

size_t QnnAOTDataTypeSize(Qnn_DataType_t dtype);

// Collection of symbols that we need to load from qnn dyn lib.
struct QnnFuncSymbols {
  using QnnInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnInterface_t*** providerList, uint32_t* numProviders);
  using QnnSystemInterfaceGetProvidersFuncType = Qnn_ErrorHandle_t(const QnnSystemInterface_t*** providerList,
                                                                   uint32_t* numProviders);

  QNN_INTERFACE_VER_TYPE qnn_interface_;
  QNN_SYSTEM_INTERFACE_VER_TYPE qnn_system_interface_;
};

class QnnAOTParamScalar {
 public:
  using ptr_t = std::shared_ptr<QnnAOTParamScalar>;

  template<typename T>
  static inline ptr_t create(const std::string& name, T value) {
    return std::make_shared<QnnAOTParamScalar>(name, value);
  };

  QnnAOTParamScalar(const std::string& name, bool value);

  QnnAOTParamScalar(const std::string& name, uint32_t value);

  QnnAOTParamScalar(const std::string& name, float value);

  Qnn_Param_t* getQnnParam();

 private:
  std::string name_;
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

class QnnAOTParamTensor {
 public:
  using ptr_t = std::shared_ptr<QnnAOTParamTensor>;

  static inline ptr_t create(const std::string& param_name, const std::string& tensor_name, Qnn_DataType_t data_type,
                             const std::vector<int32_t>& dimensions) {
    std::vector<uint32_t> vec(dimensions.size());
    for (int i = 0; i < dimensions.size(); i++) { vec[i] = (uint32_t)dimensions[i]; }
    return std::make_shared<QnnAOTParamTensor>(param_name, tensor_name, data_type, vec);
  }

  static inline ptr_t create(const std::string& param_name, const std::string& tensor_name, Qnn_DataType_t data_type,
                             const std::vector<uint32_t>& dimensions) {
    return std::make_shared<QnnAOTParamTensor>(param_name, tensor_name, data_type, dimensions);
  }

  QnnAOTParamTensor(const std::string& param_name, const std::string& tensor_name, Qnn_DataType_t data_type,
                    const std::vector<uint32_t>& dimensions);

  ~QnnAOTParamTensor();

  void* alloc();

  Qnn_Param_t* getQnnParam();

  Qnn_Tensor_t* getQnnTensor();

 private:
  std::string param_name_;
  std::string tensor_name_;
  std::vector<uint32_t> dimensions_;
  Qnn_Param_t qnn_param_ = QNN_PARAM_INIT;
};

class QnnAOTNodeTensor : public std::enable_shared_from_this<QnnAOTNodeTensor> {
 public:
  using ptr_t = std::shared_ptr<QnnAOTNodeTensor>;

  static inline ptr_t create(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight = false) {
    return std::make_shared<QnnAOTNodeTensor>(v);
  }

  explicit QnnAOTNodeTensor(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight = false);

 private:
  Qnn_TensorType_t parseQnnTensorTypeFromIR(const ir::tensor::TensorValue::ptr_t& v);

  Qnn_DataType_t parseQnnDataTypeFromIR(const ir::tensor::TensorValue::ptr_t& v);

  std::string parseQnnTensorNameFromIR(const ir::tensor::TensorValue::ptr_t& v);

  Qnn_QuantizeParams_t parseQnnQuantizeParamFromIR(const ir::tensor::TensorValue::ptr_t& v);

  Tensor mllm_tensor_;
  std::string name_;
  std::vector<uint32_t> shape_;
  Qnn_Tensor_t qnn_tensor_ = QNN_TENSOR_INIT;
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

  QnnAOTNodeOperation::ptr_t addParamScalar(const std::vector<QnnAOTParamScalar::ptr_t>& params);

  QnnAOTNodeOperation::ptr_t emplaceParamScalar(const QnnAOTParamScalar::ptr_t& param);

  QnnAOTNodeOperation::ptr_t addParamTensor(const std::vector<QnnAOTParamTensor::ptr_t>& params);

  QnnAOTNodeOperation::ptr_t emplaceParamTensor(const QnnAOTParamTensor::ptr_t& param);

  QnnAOTNodeOperation::ptr_t setOpName(const std::string& op_name);

  QnnAOTNodeOperation::ptr_t setName(const std::string& name);

  std::string getName();

  QnnAOTNodeOperation::ptr_t setPackageName(const std::string& package_name);

  std::string name_;
  std::string op_name_;
  std::string package_name_ = "qti.aisw";
  std::vector<QnnAOTParamScalar::ptr_t> param_scalar;
  std::vector<QnnAOTParamTensor::ptr_t> param_tensor;
  std::vector<QnnAOTNodeTensor::ptr_t> inputs;
  std::vector<QnnAOTNodeTensor::ptr_t> outputs;
};

class QnnAOTGraph : public std::enable_shared_from_this<QnnAOTGraph> {
 public:
  using ptr_t = std::shared_ptr<QnnAOTGraph>;

  void addOperation(const QnnAOTNodeOperation::ptr_t& qnn_op);

  bool compile();

  bool is_compiled_ = false;
  std::unordered_map<std::string, QnnAOTNodeOperation::ptr_t> op_node_;
  std::unordered_map<std::string, QnnAOTNodeTensor::ptr_t> all_tensors_;

 private:
  std::string graph_name_;
  std::string belongs_context_name_;
  Qnn_GraphHandle_t qnn_graph_handle_ = nullptr;
};

struct QnnDeviceAndContext {
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
