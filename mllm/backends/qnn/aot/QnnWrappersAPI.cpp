// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#include <memory>

#include <QNN/QnnTypes.h>

#include <QNN/QnnGraph.h>
#include <QNN/QnnContext.h>
#include <QNN/HTP/QnnHtpDevice.h>
#include <QNN/HTP/QnnHtpCommon.h>
#include <QNN/HTP/QnnHtpContext.h>

#include "mllm/utils/Common.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/backends/qnn/QNNTypeMacros.hpp"
#include "mllm/compile/ir/linalg/Attribute.hpp"
#include "mllm/backends/qnn/aot/QnnWrappersAPI.hpp"
#include "mllm/backends/qnn/aot/QnnTargetMachine.hpp"
#include "mllm/backends/qnn/aot/passes/AOTCompileContext.hpp"

namespace mllm::qnn::aot {

void __mllmLoggerCallback4QnnLogger(const char* fmt, QnnLog_Level_t level, uint64_t times_tamp, va_list argp) {
  const char* level_str = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR: level_str = "[ERROR]  "; break;
    case QNN_LOG_LEVEL_WARN: level_str = "[WARN]   "; break;
    case QNN_LOG_LEVEL_INFO: level_str = "[INFO]   "; break;
    case QNN_LOG_LEVEL_DEBUG: level_str = "[DEBUG]  "; break;
    case QNN_LOG_LEVEL_VERBOSE: level_str = "[VERBOSE]"; break;
    case QNN_LOG_LEVEL_MAX: level_str = "[UNKNOWN]"; break;
  }

  double ms = (double)times_tamp / 1000000.0;

  {
    fprintf(stdout, "QnnLogger(%8.1fms, %ld) %s: ", ms, times_tamp, level_str);
    vfprintf(stdout, fmt, argp);
  }
}

size_t QnnAOTDataTypeSize(Qnn_DataType_t dtype) {
  switch (dtype) {
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_BOOL_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8: return 1;

    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16: return 2;

    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32: return 4;

    case QNN_DATATYPE_INT_64:
    case QNN_DATATYPE_UINT_64: return 8;

    default:
      MLLM_ERROR("QnnAOTDataTypeSize: unsupported Qnn_DataType_t {}", static_cast<int>(dtype));
      MLLM_RT_ASSERT(false);
      return 0;
  }
}

QnnAOTParamScalar::QnnAOTParamScalar(const std::string& name, bool value) {
  name_ = name;
  qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
  qnn_param_.name = name_.c_str();
  qnn_param_.scalarParam.dataType = QNN_DATATYPE_BOOL_8;
  qnn_param_.scalarParam.bool8Value = static_cast<uint8_t>(value);
}

QnnAOTParamScalar::QnnAOTParamScalar(const std::string& name, uint32_t value) {
  name_ = name;
  qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
  qnn_param_.name = name_.c_str();
  qnn_param_.scalarParam.dataType = QNN_DATATYPE_UINT_32;
  qnn_param_.scalarParam.uint32Value = value;
}

QnnAOTParamScalar::QnnAOTParamScalar(const std::string& name, float value) {
  name_ = name;
  qnn_param_.paramType = QNN_PARAMTYPE_SCALAR;
  qnn_param_.name = name_.c_str();
  qnn_param_.scalarParam.dataType = QNN_DATATYPE_FLOAT_32;
  qnn_param_.scalarParam.floatValue = value;
}

Qnn_Param_t* QnnAOTParamScalar::getQnnParam() { return &(qnn_param_); }

QnnAOTParamTensor::QnnAOTParamTensor(const std::string& param_name, const std::string& tensor_name, Qnn_DataType_t data_type,
                                     const std::vector<uint32_t>& dimensions) {
  param_name_ = param_name;
  tensor_name_ = tensor_name;
  dimensions_ = dimensions;
  // Fix parameters.
  qnn_param_.paramType = QNN_PARAMTYPE_TENSOR;
  qnn_param_.tensorParam.version = QNN_TENSOR_VERSION_2;
  qnn_param_.tensorParam.v2 = QNN_TENSOR_V2_INIT;
  qnn_param_.tensorParam.v2.type = QNN_TENSOR_TYPE_STATIC;
  qnn_param_.tensorParam.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qnn_param_.tensorParam.v2.quantizeParams = Qnn_QuantizeParams_t{
      QNN_DEFINITION_UNDEFINED, QNN_QUANTIZATION_ENCODING_UNDEFINED, {.scaleOffsetEncoding = {.scale = 0.0f, .offset = 0}}};
  qnn_param_.tensorParam.v2.memType = QNN_TENSORMEMTYPE_RAW;
  // Custom parameters.
  qnn_param_.name = param_name_.c_str();
  qnn_param_.tensorParam.v2.name = tensor_name_.c_str();
  qnn_param_.tensorParam.v2.dataType = data_type;
  qnn_param_.tensorParam.v2.rank = dimensions_.size();
  qnn_param_.tensorParam.v2.dimensions = dimensions_.data();
  qnn_param_.tensorParam.v2.clientBuf = {.data = nullptr, .dataSize = 0};
}

QnnAOTParamTensor::~QnnAOTParamTensor() {
  auto data = QNN_TENSOR_GET_CLIENT_BUF(qnn_param_.tensorParam).data;
  MLLM_RT_ASSERT(data != nullptr);
  if (data) { free(data); }
}

void* QnnAOTParamTensor::alloc() {
  uint32_t data_size = QnnAOTDataTypeSize(QNN_TENSOR_GET_DATA_TYPE(qnn_param_.tensorParam));
  for (int i = 0; i < QNN_TENSOR_GET_RANK(qnn_param_.tensorParam); i++) {
    data_size *= qnn_param_.tensorParam.v2.dimensions[i];
  }
  Qnn_ClientBuffer_t clientBuffer = {.data = malloc(data_size), .dataSize = data_size};
  QNN_TENSOR_SET_CLIENT_BUF(qnn_param_.tensorParam, clientBuffer);
  MLLM_RT_ASSERT(QNN_TENSOR_GET_CLIENT_BUF(qnn_param_.tensorParam).data != nullptr);
  return QNN_TENSOR_GET_CLIENT_BUF(qnn_param_.tensorParam).data;
}

Qnn_Param_t* QnnAOTParamTensor::getQnnParam() { return &qnn_param_; }

Qnn_Tensor_t* QnnAOTParamTensor::getQnnTensor() { return &qnn_param_.tensorParam; }

QnnAOTNodeTensor::QnnAOTNodeTensor(const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight) {
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned

  name_ = v->name();
  mllm_tensor_ = v->tensor_;
  quant_spec_ = v->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;
  for (auto s : v->tensor_.shape()) { shape_.emplace_back(s); }

  qnn_tensor_.version = QNN_TENSOR_VERSION_2;
  qnn_tensor_.v2 = QNN_TENSOR_V2_INIT;
  qnn_tensor_.v2.id = v->tensor_.uuid();
  qnn_tensor_.v2.name = name_.c_str();
  qnn_tensor_.v2.type = parseQnnTensorTypeFromIR(v);
  qnn_tensor_.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  qnn_tensor_.v2.dataType = parseQnnDataTypeFromIR(v);
  qnn_tensor_.v2.quantizeParams = parseQnnQuantizeParamFromIR(v);
  qnn_tensor_.v2.rank = (uint32_t)v->tensor_.rank();
  qnn_tensor_.v2.dimensions = shape_.data();
  qnn_tensor_.v2.isDynamicDimensions = nullptr;
  qnn_tensor_.v2.sparseParams = QNN_SPARSE_PARAMS_INIT;
  qnn_tensor_.v2.isProduced = 0u;

  if (force_static_weight) {
    qnn_tensor_.v2.memType = QNN_TENSORMEMTYPE_RAW;
    qnn_tensor_.v2.clientBuf = {
        .data = (void*)mllm_tensor_.ptr<char>(),
        .dataSize = (uint32_t)mllm_tensor_.bytes(),
    };
  }
}

Qnn_TensorType_t QnnAOTNodeTensor::parseQnnTensorTypeFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  auto type = v->tensor_.memType();
  Qnn_TensorType_t ret_qnn_tensor_type = QNN_TENSOR_TYPE_UNDEFINED;
  switch (type) {
    case kTensorMemTypes_Start: {
      break;
    }

    // For MLLM Frame work to use
    case kNormal: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_NATIVE;
      break;
    }
    case kExtraInput: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
      break;
    }
    case kExtraOutput: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
      break;
    }
    case kManual: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
      break;
    }
    case kGlobal: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC;
      break;
    }

    // Framework need to judge if this tensor is mmap from disk.
    case kParams_Start:
    case kParamsMMAP:
    case kParamsNormal:
    case kParams_End: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_STATIC;
      break;
    }

    // For QNN Backend to use.
    case kQnnAppRead: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ;
      break;
    }
    case kQnnAppWrite: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_WRITE;
      break;
    }
    case kQnnAppReadWrite: {
      ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE;
      break;
    }
    case kTensorMemTypes_End: break;
  }

  // Check Attribute. The Attribute priority is higher than tensor type
  if (v->getAttr("qnn_graph_outputs")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READ; }
  if (v->getAttr("qnn_graph_inputs")) { ret_qnn_tensor_type = QNN_TENSOR_TYPE_APP_READWRITE; }

  return ret_qnn_tensor_type;
}

Qnn_DataType_t QnnAOTNodeTensor::parseQnnDataTypeFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  Qnn_DataType_t ret = QNN_DATATYPE_UNDEFINED;
  switch (v->tensor_.dtype()) {
    case kInt8: {
      ret = QNN_DATATYPE_INT_8;
      break;
    }
    case kInt16: {
      ret = QNN_DATATYPE_INT_16;
      break;
    }
    case kInt32: {
      ret = QNN_DATATYPE_INT_32;
      break;
    }
    case kInt64: {
      ret = QNN_DATATYPE_INT_64;
      break;
    }
    case kUInt8: {
      ret = QNN_DATATYPE_UINT_8;
      break;
    }
    case kUInt16: {
      ret = QNN_DATATYPE_UINT_16;
      break;
    }
    case kUInt32: {
      ret = QNN_DATATYPE_UINT_32;
      break;
    }
    case kUInt64: {
      ret = QNN_DATATYPE_UINT_64;
      break;
    }
    case kFloat16: {
      ret = QNN_DATATYPE_FLOAT_16;
      break;
    }
    case kFloat32: {
      ret = QNN_DATATYPE_FLOAT_32;
      break;
    }
    case kBFloat16: {
      ret = QNN_DATATYPE_BFLOAT_16;
      break;
    }
    // FIXME: Maybe error here.
    case kInt4: {
      ret = QNN_DATATYPE_SFIXED_POINT_4;
      break;
    }
    case kUInt4: {
      ret = QNN_DATATYPE_UFIXED_POINT_4;
      break;
    }
    case kInt8PerTensorSym:
    case kInt8PerTensorAsy:
    case kInt8PerChannelAsy:
    case kInt8PerChannelSym: {
      ret = QNN_DATATYPE_SFIXED_POINT_8;
      break;
    }
    case kUInt8PerTensorSym:
    case kUInt8PerTensorAsy:
    case kUInt8PerChannelAsy:
    case kUInt8PerChannelSym: {
      ret = QNN_DATATYPE_UFIXED_POINT_8;
      break;
    }
    case kInt16PerTensorSym:
    case kInt16PerTensorAsy:
    case kInt16PerChannelSym:
    case kInt16PerChannelAsy: {
      ret = QNN_DATATYPE_SFIXED_POINT_16;
      break;
    }
    case kUInt16PerTensorSym:
    case kUInt16PerTensorAsy:
    case kUInt16PerChannelSym:
    case kUInt16PerChannelAsy: {
      ret = QNN_DATATYPE_UFIXED_POINT_16;
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't parse datatype: {}", nameOfType(v->tensor_.dtype()));
      ret = QNN_DATATYPE_UNDEFINED;
    }
  }
  return ret;
}

std::string QnnAOTNodeTensor::parseQnnTensorNameFromIR(const ir::tensor::TensorValue::ptr_t& v) { return v->name(); }

Qnn_QuantizeParams_t QnnAOTNodeTensor::parseQnnQuantizeParamFromIR(const ir::tensor::TensorValue::ptr_t& v) {
  Qnn_QuantizeParams_t ret = QNN_QUANTIZE_PARAMS_INIT;

  MLLM_RT_ASSERT(v->getAttr("quant_recipe"));
  auto quant_spec = v->getAttr("quant_recipe")->cast_<ir::linalg::LinalgIRQuantizatonSpecAttr>()->spec_;

  switch (quant_spec->type) {
    case ir::linalg::QuantizationSpecType::kRaw: {
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerTensor: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerTensor>(quant_spec);
      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
      ret.scaleOffsetEncoding = Qnn_ScaleOffset_t{.scale = cfg->scale.item<float>(), .offset = 0};
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerChannel: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecSymPerChannel>(quant_spec);

      // Prepare data
      auto num_scale_offsets = (uint32_t)v->tensor_.size(cfg->ch_axis);
      Qnn_ScaleOffset_t* scale_array = (Qnn_ScaleOffset_t*)malloc(sizeof(Qnn_ScaleOffset_t) * num_scale_offsets);
      MLLM_RT_ASSERT_EQ(num_scale_offsets, cfg->scale.size(0));
      MLLM_RT_ASSERT_EQ(cfg->scale.dtype(), kFloat32);
      for (int i = 0; i < num_scale_offsets; ++i) {
        scale_array[i].scale = cfg->scale.at<float>({i});
        scale_array[i].offset = 0;
      }
      unreachable_handle_.emplace_back(scale_array);

      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET;
      ret.axisScaleOffsetEncoding = Qnn_AxisScaleOffset_t{
          .axis = cfg->ch_axis,
          .numScaleOffsets = num_scale_offsets,
          .scaleOffset = scale_array,
      };
      break;
    }
    case ir::linalg::QuantizationSpecType::kSymPerBlock:
    case ir::linalg::QuantizationSpecType::kAsymPerTensor:
    case ir::linalg::QuantizationSpecType::kAsymPerChannel:
    case ir::linalg::QuantizationSpecType::kAsymPerBlock: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't handle [kSymPerBlock, kAsymPerTensor, kAsymPerChannel, kAsymPerBlock] type");
    }
    case ir::linalg::QuantizationSpecType::kLPBQ: {
      auto cfg = std::static_pointer_cast<ir::linalg::QuantizationSpecLPBQ>(quant_spec);

      // Prepare data
      auto num_scale_offsets = (uint32_t)v->tensor_.size(cfg->ch_axis);
      Qnn_ScaleOffset_t* scale_array = (Qnn_ScaleOffset_t*)malloc(sizeof(Qnn_ScaleOffset_t) * num_scale_offsets);
      MLLM_RT_ASSERT_EQ(num_scale_offsets, cfg->scale_level_1_fp.size(0));
      MLLM_RT_ASSERT_EQ(cfg->scale_level_0_int.dtype(), kUInt8);
      for (int i = 0; i < num_scale_offsets; ++i) {
        scale_array[i].scale = cfg->scale_level_1_fp.at<float>({i});
        scale_array[i].offset = 0;
      }
      unreachable_handle_.emplace_back(scale_array);

      auto block_scale_array = (Qnn_BlockwiseExpansion_t*)malloc(sizeof(Qnn_BlockwiseExpansion_t));
      unreachable_handle_.emplace_back(block_scale_array);
      block_scale_array[0].axis = cfg->ch_axis;
      block_scale_array[0].scaleOffsets = scale_array;
      block_scale_array[0].numBlocksPerAxis = v->tensor_.size(cfg->ch_axis) / cfg->block_size;
      block_scale_array[0].blockScaleBitwidth = 12;  // 12 bits for 4 to 16 expansion
      block_scale_array[0].blockScaleStorageType = QNN_BLOCKWISE_EXPANSION_BITWIDTH_SCALE_STORAGE_8;
      block_scale_array[0].blocksScale8 = cfg->scale_level_0_int.ptr<mllm_uint8_t>();

      ret.encodingDefinition = QNN_DEFINITION_DEFINED;
      ret.quantizationEncoding = QNN_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION;
      ret.blockwiseExpansion = block_scale_array;
      break;
    }
    default: {
      MLLM_ERROR_EXIT(ExitCode::kCoreError, "Can't handle kNone type");
    }
  }

  return ret;
}

// QnnAOTNodeOperation implementations
QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addInputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ins) {
  inputs.insert(inputs.end(), ins.begin(), ins.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addOutputs(const std::vector<QnnAOTNodeTensor::ptr_t>& ous) {
  outputs.insert(outputs.end(), ous.begin(), ous.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceInput(const QnnAOTNodeTensor::ptr_t& input) {
  inputs.push_back(input);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceOutput(const QnnAOTNodeTensor::ptr_t& output) {
  outputs.push_back(output);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamScalar(const std::vector<QnnAOTParamScalar::ptr_t>& params) {
  param_scalar.insert(param_scalar.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamScalar(const QnnAOTParamScalar::ptr_t& param) {
  param_scalar.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::addParamTensor(const std::vector<QnnAOTParamTensor::ptr_t>& params) {
  param_tensor.insert(param_tensor.end(), params.begin(), params.end());
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::emplaceParamTensor(const QnnAOTParamTensor::ptr_t& param) {
  param_tensor.push_back(param);
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setOpName(const std::string& op_name) {
  op_name_ = op_name;
  return shared_from_this();
}

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setName(const std::string& name) {
  name_ = name;
  return shared_from_this();
}

std::string QnnAOTNodeOperation::getName() { return name_; }

QnnAOTNodeOperation::ptr_t QnnAOTNodeOperation::setPackageName(const std::string& package_name) {
  package_name_ = package_name;
  return shared_from_this();
}

QnnAOTGraph::QnnAOTGraph(const std::string& g_name, const std::shared_ptr<QnnDeviceAndContext>& context)
    : graph_name_(g_name), qnn_context_(context) {
  belongs_context_name_ = context->name_;

  auto env = AOTCompileContext::getInstance().getEnv();
  auto qnn_interface = env->getFuncSymbol().qnn_interface_;

  auto ok = qnn_interface.graphCreate(context->qnn_ctx_handle_, g_name.c_str(), nullptr /*graph_config*/, &qnn_graph_handle_);
  MLLM_RT_ASSERT_EQ(ok, QNN_SUCCESS);
}

void QnnAOTGraph::addOperation(const QnnAOTNodeOperation::ptr_t& qnn_op) {
  auto env = AOTCompileContext::getInstance().getEnv();
  auto qnn_interface = env->getFuncSymbol().qnn_interface_;

  Qnn_OpConfig_t qnn_op_config = QNN_OPCONFIG_INIT;
  qnn_op_config.version = QNN_OPCONFIG_VERSION_1;
  qnn_op_config.v1 = QNN_OPCONFIG_V1_INIT;
  qnn_op_config.v1.name = qnn_op->name_.c_str();
  qnn_op_config.v1.packageName = qnn_op->package_name_.c_str();
  qnn_op_config.v1.typeName = qnn_op->op_name_.c_str();

  // TODO PARAMs
  // TODO Inputs
  // TODO Outputs

  // TODO node validations

  // TODO add node to graph.

  op_node_.insert({qnn_op->getName(), qnn_op});
}

bool QnnAOTGraph::compile() {
  if (is_compiled_) { return true; }
  // TODO

  is_compiled_ = true;
  return true;
}

const std::vector<std::string> QnnDynSymbolLoader::possible_qnn_dyn_lib_paths_{
    "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/",
};

QnnDynSymbolLoader::~QnnDynSymbolLoader() {
  for (auto& item : libs_) {
    if (item.second.handle_) { dlclose(item.second.handle_); }
  }
}

bool QnnDynSymbolLoader::loadQnnDynLib(const std::string& lib_name, int flag) {
  for (auto const& path : possible_qnn_dyn_lib_paths_) {
    auto real_path = path + lib_name;
    auto handle = dlopen(real_path.c_str(), flag);
    if (handle) {
      auto descriptor = QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
      libs_.insert({lib_name, descriptor});
      MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
      return true;
    } else {
      char* error = dlerror();
      MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib try for {} failed: {}", real_path, error ? error : "Unknown error");
    }
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

bool QnnDynSymbolLoader::loadQnnDynLibAtPath(const std::string& path, const std::string& lib_name, int flag) {
  auto real_path = path + lib_name;
  auto handle = dlopen(real_path.c_str(), flag);
  if (handle) {
    auto descriptor = QnnDynLibDescriptor{.lib_name_ = lib_name, .lib_path_ = path, .handle_ = handle};
    libs_.insert({lib_name, descriptor});
    MLLM_INFO("QnnDynSymbolLoader::loadQnnDynLib {} success.", real_path);
    return true;
  } else {
    char* error = dlerror();
    MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib try for {} failed: {}", real_path, error ? error : "Unknown error");
  }
  MLLM_ERROR("QnnDynSymbolLoader::loadQnnDynLib {} failed.", lib_name);
  return false;
}

QnnAOTEnv::QnnAOTEnv(const QcomTargetMachine& target_machine) : target_machine_(target_machine) { _setup(); }

QnnAOTEnv::QnnAOTEnv(const std::string& lib_path, const QcomTargetMachine& target_machine) : target_machine_(target_machine) {
  _setup(lib_path);
}

void QnnAOTEnv::_setup(const std::string& path) {
  auto& loader = QnnDynSymbolLoader::instance();
  std::string htp_backend_lib_name = "libQnnHtp.so";
  // GLOBAL Load
  if (path.empty()) {
    if (!loader.loadQnnDynLib(htp_backend_lib_name,
                              QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
      MLLM_ERROR("QnnAOTEnv::QnnAOTEnv {} failed.", htp_backend_lib_name);
      exit(1);
    }
  } else {
    if (!loader.loadQnnDynLibAtPath(path, htp_backend_lib_name,
                                    QnnDynSymbolLoader::DynFlag::kRTLD_NOW | QnnDynSymbolLoader::DynFlag::kRTLD_GLOBAL)) {
      MLLM_ERROR("QnnAOTEnv::QnnAOTEnv {} failed.", htp_backend_lib_name);
      exit(1);
    }
  }

  auto qnn_interface_get_providers_func =
      loader(htp_backend_lib_name).func<QnnFuncSymbols::QnnInterfaceGetProvidersFuncType>("QnnInterface_getProviders");

  QnnInterface_t** interface_providers = nullptr;
  uint32_t num_providers = 0;

  MLLM_RT_ASSERT_EQ(qnn_interface_get_providers_func((const QnnInterface_t***)&interface_providers, &num_providers),
                    QNN_SUCCESS);
  MLLM_RT_ASSERT(interface_providers != nullptr);
  MLLM_RT_ASSERT(num_providers != 0);

  MLLM_INFO("QnnAOTEnv::QnnAOTEnv get HTP num_providers: {}", num_providers);

  bool found_valid_interface = false;
  // Get correct provider
  for (size_t provider_id = 0; provider_id < num_providers; provider_id++) {
    if (QNN_API_VERSION_MAJOR == interface_providers[provider_id]->apiVersion.coreApiVersion.major
        && QNN_API_VERSION_MINOR <= interface_providers[provider_id]->apiVersion.coreApiVersion.minor) {
      found_valid_interface = true;
      qnn_htp_func_symbols_.qnn_interface_ = interface_providers[provider_id]->QNN_INTERFACE_VER_NAME;
      break;
    }
  }
  MLLM_RT_ASSERT_EQ(found_valid_interface, true);

  // Check if this HTP Backend has specific property
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.propertyHasCapability(QNN_PROPERTY_GROUP_DEVICE);
    if (status == QNN_PROPERTY_NOT_SUPPORTED) { MLLM_WARN("Device property is not supported"); }

    MLLM_RT_ASSERT(status != QNN_PROPERTY_ERROR_UNKNOWN_KEY);
  }

  // Try to config this target machine
  {
    auto device_custom_config = createDecideCustomConfigInfo();
    QnnHtpDevice_CustomConfig_t* p_custom_config = nullptr;

    switch (target_machine_.soc_htp_security_pd_session) {
      case QcomSecurityPDSession::kHtpSignedPd: {
        p_custom_config = (QnnHtpDevice_CustomConfig_t*)malloc(sizeof(QnnHtpDevice_CustomConfig_t));
        unreachable_handle_.push_back(p_custom_config);
        p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SIGNEDPD;
        p_custom_config->useSignedProcessDomain.useSignedProcessDomain = true;
        p_custom_config->useSignedProcessDomain.deviceId = 0;
        device_custom_config.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));
        break;
      }
      case QcomSecurityPDSession::kHtpUnsignedPd:
      default: break;
    }

    const std::vector<QnnDevice_PlatformInfo_t*> device_platform_info = createDevicePlatformInfo();
    uint32_t num_custom_configs = device_platform_info.size() + device_custom_config.size();
    target_machine_qnn_config_.resize(num_custom_configs);

    for (std::size_t i = 0; i < device_custom_config.size(); ++i) {
      target_machine_qnn_config_[i].option = QNN_DEVICE_CONFIG_OPTION_CUSTOM;
      target_machine_qnn_config_[i].customConfig = device_custom_config[i];
      target_machine_qnn_config_ptrs_.push_back(&target_machine_qnn_config_[i]);
    }

    if (!device_platform_info.empty()) {
      // The length of platform info can only be 1.
      MLLM_RT_ASSERT_EQ(device_platform_info.size(), 1u);
      target_machine_qnn_config_[device_custom_config.size()].option = QNN_DEVICE_CONFIG_OPTION_PLATFORM_INFO;
      target_machine_qnn_config_[device_custom_config.size()].hardwareInfo = device_platform_info.back();
      target_machine_qnn_config_ptrs_.push_back(&target_machine_qnn_config_[device_custom_config.size()]);
    }

    // null terminated
    target_machine_qnn_config_ptrs_.push_back(nullptr);
  }
}

std::shared_ptr<QnnDeviceAndContext> QnnAOTEnv::createContext(const std::string& name, bool weights_sharing) {
  std::shared_ptr<QnnDeviceAndContext> context = std::make_shared<QnnDeviceAndContext>();
  context->name_ = name;

  // 1. create logger and register callback.
  // clang-format off
  MLLM_RT_ASSERT_EQ(qnn_htp_func_symbols_.qnn_interface_.logCreate(__mllmLoggerCallback4QnnLogger,QNN_LOG_LEVEL_VERBOSE, &context->log_), QNN_SUCCESS)
  MLLM_RT_ASSERT_EQ(QNN_BACKEND_NO_ERROR, qnn_htp_func_symbols_.qnn_interface_.backendCreate(context->log_, (const QnnBackend_Config_t**)context->bk_cfg_, &context->bk_handle_))
  // clang-format on

  // 2. Create HTP Device
  // clang-format off
  if (nullptr != qnn_htp_func_symbols_.qnn_interface_.deviceCreate) {
    auto status = qnn_htp_func_symbols_.qnn_interface_.deviceCreate(context->log_, target_machine_qnn_config_ptrs_.data(), &context->device_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
  }
  // clang-format on

  // 3. Create Profile
  {
    auto status = qnn_htp_func_symbols_.qnn_interface_.profileCreate(context->bk_handle_, QNN_PROFILE_LEVEL_DETAILED,
                                                                     &context->profile_bk_handle_);
    MLLM_RT_ASSERT_EQ(status, QNN_SUCCESS);
  }

  // 4. Create Context
  {
    auto cfgs = createContextCustomConfig(weights_sharing);
    if (cfgs.size()) {
      context->qnn_context_config_ = (QnnContext_Config_t**)malloc(sizeof(QnnContext_Config_t*) * (cfgs.size() + 1));
      unreachable_handle_.emplace_back(context->qnn_context_config_);
    }
    for (int i = 0; i < cfgs.size(); ++i) {
      context->qnn_context_config_[i] = (QnnContext_Config_t*)malloc(sizeof(QnnContext_Config_t));
      context->qnn_context_config_[i]->option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
      context->qnn_context_config_[i]->customConfig = cfgs[i];
      unreachable_handle_.emplace_back(context->qnn_context_config_[i]);
    }
    if (cfgs.size()) { context->qnn_context_config_[cfgs.size()] = nullptr; }
    auto status = qnn_htp_func_symbols_.qnn_interface_.contextCreate(context->bk_handle_, context->device_handle_,
                                                                     (const QnnContext_Config_t**)context->qnn_context_config_,
                                                                     &context->qnn_ctx_handle_);
    MLLM_RT_ASSERT_EQ(QNN_CONTEXT_NO_ERROR, status);
  }

  // 5. Register MLLM's Qnn Opset
  // clang-format off
  {
    // FIXME(wch): we need to register our own opset of qnn.
  }
  // clang-format on

  MLLM_RT_ASSERT_EQ(contexts_.count(name), 0);
  contexts_[name] = context;
  return context;
}

void QnnAOTEnv::saveContext(const std::string& name, const std::string& path) {
  // TODO
}

void QnnAOTEnv::destroyContext(const std::string& name) {
  // TODO
}

std::vector<QnnDevice_PlatformInfo_t*> QnnAOTEnv::createDevicePlatformInfo() {
  std::vector<QnnDevice_PlatformInfo_t*> ret;
  QnnDevice_PlatformInfo_t* p_platform_info = nullptr;
  QnnDevice_HardwareDeviceInfo_t* p_hw_device_info = nullptr;
  QnnHtpDevice_DeviceInfoExtension_t* p_device_info_extension = nullptr;
  QnnDevice_CoreInfo_t* p_core_info = nullptr;

  p_platform_info = (QnnDevice_PlatformInfo_t*)malloc(sizeof(QnnDevice_PlatformInfo_t));
  unreachable_handle_.push_back(p_platform_info);
  p_platform_info->version = QNN_DEVICE_PLATFORM_INFO_VERSION_1;
  p_platform_info->v1.numHwDevices = 1;

  p_hw_device_info = (QnnDevice_HardwareDeviceInfo_t*)malloc(sizeof(QnnDevice_HardwareDeviceInfo_t));
  unreachable_handle_.push_back(p_hw_device_info);
  p_hw_device_info->version = QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1;
  p_hw_device_info->v1.deviceId = 0;
  p_hw_device_info->v1.deviceType = 0;
  p_hw_device_info->v1.numCores = 1;

  p_device_info_extension = (QnnHtpDevice_DeviceInfoExtension_t*)malloc(sizeof(QnnHtpDevice_DeviceInfoExtension_t));
  unreachable_handle_.push_back(p_device_info_extension);
  // clang-format off
  p_device_info_extension->devType = QNN_HTP_DEVICE_TYPE_ON_CHIP;
  p_device_info_extension->onChipDevice.vtcmSize = target_machine_.soc_htp_vtcm_total_memory_size;  // in MB
  p_device_info_extension->onChipDevice.signedPdSupport = target_machine_.soc_htp_security_pd_session == QcomSecurityPDSession::kHtpSignedPd;
  p_device_info_extension->onChipDevice.socModel = static_cast<uint32_t>(target_machine_.soc_htp_chipset);
  p_device_info_extension->onChipDevice.arch = static_cast<QnnHtpDevice_Arch_t>(target_machine_.soc_htp_arch);
  p_device_info_extension->onChipDevice.dlbcSupport = true;
  p_hw_device_info->v1.deviceInfoExtension = p_device_info_extension;
  // clang-format on

  p_core_info = (QnnDevice_CoreInfo_t*)malloc(sizeof(QnnDevice_CoreInfo_t));
  unreachable_handle_.push_back(p_core_info);
  p_core_info->version = QNN_DEVICE_CORE_INFO_VERSION_1;
  p_core_info->v1.coreId = 0;
  p_core_info->v1.coreType = 0;
  p_core_info->v1.coreInfoExtension = nullptr;
  p_hw_device_info->v1.cores = p_core_info;

  p_platform_info->v1.hwDevices = p_hw_device_info;
  ret.push_back(p_platform_info);

  return ret;
}

std::vector<QnnDevice_CustomConfig_t> QnnAOTEnv::createDecideCustomConfigInfo() {
  std::vector<QnnDevice_CustomConfig_t> ret;

  QnnHtpDevice_CustomConfig_t* p_custom_config = (QnnHtpDevice_CustomConfig_t*)malloc(sizeof(QnnHtpDevice_CustomConfig_t));
  unreachable_handle_.push_back(p_custom_config);
  p_custom_config->option = QNN_HTP_DEVICE_CONFIG_OPTION_SOC;
  p_custom_config->socModel = static_cast<uint32_t>(target_machine_.soc_htp_chipset);
  ret.push_back(static_cast<QnnDevice_CustomConfig_t>(p_custom_config));

  return ret;
}

std::vector<QnnContext_CustomConfig_t> QnnAOTEnv::createContextCustomConfig(bool weights_sharing) {
  std::vector<QnnContext_CustomConfig_t> ret;
  QnnHtpContext_CustomConfig_t* p_custom_config = nullptr;

  if (weights_sharing) {
    p_custom_config = (QnnHtpContext_CustomConfig_t*)malloc(sizeof(QnnHtpContext_CustomConfig_t));
    unreachable_handle_.push_back(p_custom_config);
    p_custom_config->option = QNN_HTP_CONTEXT_CONFIG_OPTION_WEIGHT_SHARING_ENABLED;
    p_custom_config->weightSharingEnabled = true;
    ret.push_back(static_cast<QnnContext_CustomConfig_t>(p_custom_config));
  }

  return ret;
}

QnnAOTGraph::ptr_t QnnAOTEnv::captureAOTGraph(const std::string& qnn_context_name, const std::string& g_name) {
  MLLM_RT_ASSERT(contexts_.count(qnn_context_name) == 1);
  auto ret = QnnAOTGraph::create(g_name, contexts_[qnn_context_name]);
  ret->belongs_context_name_ = qnn_context_name;
  contexts_[qnn_context_name]->graphs_.insert({g_name, ret});
  return ret;
}

void QnnAOTEnv::captureAOTNodeOp(const std::string& qnn_context_name, const std::string& graph_name,
                                 const QnnAOTNodeOperation::ptr_t& op) {
  MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
  MLLM_RT_ASSERT_EQ(contexts_[qnn_context_name]->graphs_.count(graph_name), 1);
  contexts_[qnn_context_name]->graphs_[graph_name]->addOperation(op);
}

QnnAOTNodeTensor::ptr_t QnnAOTEnv::captureQnnAOTNodeTensor(const std::string& qnn_context_name, const std::string& graph_name,
                                                           const ir::tensor::TensorValue::ptr_t& v, bool force_static_weight) {
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  // TODO Constant value should also use Static!!! And they can be pruned
  auto __qnn_tensor_name = v->name();

  bool __qnn_enable_static_weight = force_static_weight;

  // Check if this value want static qnn weight. The static qnn weight will be shared through one context in diff graphs!
  if (v->tensor_.memType() == kGlobal || (v->tensor_.memType() <= kParams_End && v->tensor_.memType() >= kParams_Start)) {
    __qnn_enable_static_weight = true;
  }

  // If static weight is cached, we return it directly.
  if (__qnn_enable_static_weight) {
    MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
    if (contexts_[qnn_context_name]->static_tensor_.count(__qnn_tensor_name)) {
      return contexts_[qnn_context_name]->static_tensor_[__qnn_tensor_name];
    }
  }

  // If normal weight is cached, we return it directly
  MLLM_RT_ASSERT_EQ(contexts_.count(qnn_context_name), 1);
  MLLM_RT_ASSERT_EQ(contexts_[qnn_context_name]->graphs_.count(graph_name), 1);
  if (contexts_[qnn_context_name]->graphs_[graph_name]->all_tensors_.count(__qnn_tensor_name)) {
    return contexts_[qnn_context_name]->graphs_[graph_name]->all_tensors_[__qnn_tensor_name];
  }

  // There has no Tensor in the cache.
  // Create Tensor and register it!.
  auto ret = QnnAOTNodeTensor::create(v, __qnn_enable_static_weight);
  if (__qnn_enable_static_weight) {
    contexts_[qnn_context_name]->static_tensor_.insert({__qnn_tensor_name, ret});
  } else {
    contexts_[qnn_context_name]->graphs_[graph_name]->all_tensors_.insert({__qnn_tensor_name, ret});
  }

  return ret;
}

}  // namespace mllm::qnn::aot
