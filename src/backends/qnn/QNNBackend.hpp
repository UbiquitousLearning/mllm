#ifndef MLLM_QNNBACKEND_H
#define MLLM_QNNBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "OpDefined.hpp"
#include "ParamLoader.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "MemoryManager.hpp"
#include <memory>

#include "Utils/IOTensor.hpp"
#include "PAL/DynamicLoading.hpp"
#include "Model/QnnModel.hpp"
#include "QNN.hpp"
#include "Log/Logger.hpp"

using std::shared_ptr;

using namespace qnn;
using namespace qnn::tools;

namespace mllm {

enum class StatusCode {
    SUCCESS,
    FAILURE,
    FAILURE_INPUT_LIST_EXHAUSTED,
    FAILURE_SYSTEM_ERROR,
    FAILURE_SYSTEM_COMMUNICATION_ERROR,
    QNN_FEATURE_UNSUPPORTED
};

class Op;

class Tensor;
class Backend;
class QNNBackend : public Backend {
public:
    QNNBackend(shared_ptr<MemoryManager> mm);
    ~QNNBackend();

    Op *opCreate(const OpParam &op_param, string name = "", int threadCount = 4) override {
        OpType optype = OpType(op_param.find("type")->second);
        auto iter = map_creator_.find(optype);
        if (iter == map_creator_.end()) {
            std::cout << "NPU Op Don't support type : " << name << std::endl;
            return nullptr;
        }
        Op *exe = nullptr;
        exe = iter->second->create(op_param, this, name);
        return exe;
    }

    // currently, qnn don't support tensor function
    TensorFunction *funcCreate(const TensorFuncType type) override {
        return nullptr;
    }

    class Creator {
    public:
        virtual ~Creator() = default;
        virtual Op *create(OpParam op_param, Backend *bn, string name) const = 0;
    };
    bool addCreator(OpType t, Creator *c) {
        if (map_creator_.find(t) != map_creator_.end()) {
            printf("Error: %d type has be added\n", t);
            return false;
        }
        map_creator_.insert(std::make_pair(t, c));
        return true;
    }

    qnn_wrapper_api::ModelError_t graphAddNode(string name, string nodeType,
                                               std::vector<string> inputTensorNames, std::vector<Qnn_Tensor_t> outputTensors,
                                               std::vector<Qnn_Param_t> params,
                                               string packageName);

    qnn_wrapper_api::ModelError_t modelAddTensor(std::string nodeName, Qnn_Tensor_t tensor);

    virtual void onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) override;
    virtual void onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) override;
    virtual void onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName = "") override;
    virtual void onExecuteEnd(std::vector<shared_ptr<Tensor>> &outputs, const string &graph_name) override;

    void freeGraphDataStructure(string graphName);

    void afterAllGraphsExecute();

    void pushInputBuffers(uint8_t *ptr) {
        currentInputBuffers->push_back(ptr);
    }
    void pushOutputBuffers(uint8_t *ptr) {
        currentOutputBuffers->push_back(ptr);
    }

    void setDataLoader(AbstructLoader *dataLoader) {
        dataLoader_ = dataLoader;
    }

private:
    qnn_wrapper_api::ModelError_t graphFinilize();
    qnn_wrapper_api::ModelError_t graphConfig();

    void registerOps() override;
    void registerFuncs() override {};

    // @brief Print a message to STDERR then exit with a non-zero
    void reportError(const std::string &err);

    StatusCode createContext();

    StatusCode registerOpPackages();

    StatusCode freeContext();

    StatusCode terminateBackend();

    StatusCode initializeProfiling();

    std::string getBackendBuildId();

    StatusCode isDevicePropertySupported();

    StatusCode createDevice();

    StatusCode freeDevice();

    StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

    StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

    StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

    StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

    AbstructLoader *dataLoader_;

    std::map<std::string, std::vector<uint8_t *>> inputBufferMap;
    std::vector<uint8_t *> *currentInputBuffers;
    std::map<std::string, std::vector<uint8_t *>> outputBufferMap;
    std::vector<uint8_t *> *currentOutputBuffers;

    std::map<OpType, QNNBackend::Creator *> map_creator_;

    std::map<std::string, int> qnnModelIndexMap_;
    std::vector<qnn_wrapper_api::QnnModel> qnnModels_;
    int qnnModelIndex_;

    sample_app::QnnFunctionPointers m_qnnFunctionPointers;

    std::vector<std::string> m_opPackagePaths;

    QnnBackend_Config_t **m_backendConfig = nullptr;
    Qnn_ContextHandle_t m_context = nullptr;
    QnnContext_Config_t **m_contextConfig = nullptr;
    bool m_debug;

    iotensor::InputDataType m_inputDataType;
    sample_app::ProfilingLevel m_profilingLevel;

    std::map<int, qnn_wrapper_api::GraphInfo_t *> graphInfoMap_;

    const QnnGraph_Config_t **graphConfigs = nullptr;
    // these two pointers is .so library handle
    void *m_backendLibraryHandle = nullptr;

    iotensor::IOTensor m_ioTensor;
    bool m_isBackendInitialized;
    bool m_isContextCreated;
    Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
    qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
    uint32_t m_graphConfigsInfoCount;
    Qnn_LogHandle_t m_logHandle = nullptr;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t m_deviceHandle = nullptr;

    std::map<int, Qnn_Tensor_t *> inputsMap_;
    std::map<int, Qnn_Tensor_t *> outputsMap_;
};

} // namespace mllm

#endif // MLLM_QNNBACKEND_H