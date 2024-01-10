#ifndef MLLM_QNNBACKEND_H
#define MLLM_QNNBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "OpDefined.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "MemoryManager.hpp"
#include "NetParameter.hpp"
#include <memory>

#include "Utils/IOTensor.hpp"
#include "PAL/DynamicLoading.hpp"
#include "QnnModel.hpp"
#include "QNN.hpp"
#include "Logger.hpp"

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
    ~QNNBackend() {
        // free creaters in map_creator_
        for (auto &iter : map_creator_) {
            delete iter.second;
        }
        // free qnn backend resource
        this->release();
        // free dynamic library handle
        if (m_backendLibraryHandle) {
            pal::dynamicloading::dlClose(m_backendLibraryHandle);
        }
        if (m_modelHandle) {
            pal::dynamicloading::dlClose(m_modelHandle);
        }
        QNN_INFO("Free handle");
    }

    Op *opCreate(const OpParam &op_param, string name = "") {
        OpType optype = OpType(op_param.find("type")->second);
        auto iter = map_creator_.find(optype);
        if (iter == map_creator_.end()) {
            printf("Don't support type \n");
            return nullptr;
        }
        Op *exe = nullptr;
        exe = iter->second->create(op_param, this, name);
        return exe;
    }
    class Creator {
    public:
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
    qnn_wrapper_api::ModelError_t graphFinilize();
    qnn_wrapper_api::ModelError_t modelAddTensor(std::string nodeName, Qnn_Tensor_t tensor);
    ErrorCode graphExecute();
    ErrorCode graphExecute(std::map<std::string, std::vector<uint8_t *>> inputBufferMap, std::map<std::string, std::vector<uint8_t *>> outputBufferMap);

    virtual void onSetUpStart(vector<shared_ptr<Tensor>> &inputs) override;
    virtual void onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) override;
    virtual void onExecuteEnd() override;

private:
    int32_t graphInitialize();

    void release();

    void registerOps();

    // @brief Print a message to STDERR then exit with a non-zero
    void reportError(const std::string &err);

    StatusCode initialize();

    StatusCode initializeBackend();

    StatusCode createContext();

    StatusCode composeGraphs();

    StatusCode finalizeGraphs();

    StatusCode executeGraphs();
    StatusCode executeGraphs(std::map<std::string, std::vector<uint8_t *>> inputBufferMap, std::map<std::string, std::vector<uint8_t *>> outputBufferMap);

    StatusCode registerOpPackages();

    StatusCode freeContext();

    StatusCode terminateBackend();

    StatusCode freeGraphs();

    Qnn_ContextHandle_t getContext();

    StatusCode initializeProfiling();

    std::string getBackendBuildId();

    StatusCode isDevicePropertySupported();

    StatusCode createDevice();

    StatusCode freeDevice();

    StatusCode verifyFailReturnStatus(Qnn_ErrorHandle_t errCode);

    StatusCode extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

    StatusCode extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

    StatusCode extractProfilingEvent(QnnProfile_EventId_t profileEventId);

    static qnn_wrapper_api::ModelError_t QnnModel_freeGraphsInfo(qnn_wrapper_api::GraphInfoPtr_t **graphsInfo, uint32_t numGraphsInfo) {
        return qnn_wrapper_api::freeGraphsInfo(graphsInfo, numGraphsInfo);
    }

    static const std::string s_defaultOutputPath;

    std::map<std::string, std::vector<uint8_t *>> inputBufferMap;
    std::vector<uint8_t *> inputBuffers;
    std::map<std::string, std::vector<uint8_t *>> outputBufferMap;
    std::vector<uint8_t *> outputBuffers;

    std::map<OpType, QNNBackend::Creator *> map_creator_;
    shared_ptr<MemoryManager> mem_manager_;
    qnn_wrapper_api::QnnModel qnnModel;

    sample_app::QnnFunctionPointers m_qnnFunctionPointers;
    std::vector<std::string> m_inputListPaths;
    std::vector<std::vector<std::queue<std::string>>> m_inputFileLists;
    std::vector<std::string> m_opPackagePaths;
    std::string m_outputPath;
    QnnBackend_Config_t **m_backendConfig = nullptr;
    Qnn_ContextHandle_t m_context = nullptr;
    QnnContext_Config_t **m_contextConfig = nullptr;
    bool m_debug;
    iotensor::OutputDataType m_outputDataType;
    iotensor::InputDataType m_inputDataType;
    sample_app::ProfilingLevel m_profilingLevel;
    bool m_dumpOutputs;
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
    // for mllm single graph execute
    qnn_wrapper_api::GraphInfo_t graphInfo;

    const QnnGraph_Config_t **graphConfigs = nullptr;
    uint32_t m_graphsCount;
    // these two pointers is .so library handle
    void *m_backendLibraryHandle = nullptr;
    void *m_modelHandle = nullptr; // m_modelHandle is always nullptr cause we build graph in runtime
    iotensor::IOTensor m_ioTensor;
    bool m_isBackendInitialized;
    bool m_isContextCreated;
    Qnn_ProfileHandle_t m_profileBackendHandle = nullptr;
    qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
    uint32_t m_graphConfigsInfoCount;
    Qnn_LogHandle_t m_logHandle = nullptr;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t m_deviceHandle = nullptr;
};

} // namespace mllm

#endif // MLLM_QNNBACKEND_H