#ifndef MLLM_QNNBACKEND_H
#define MLLM_QNNBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "Types.hpp"
#include "MemoryManager.hpp"
#include "NetParameter.hpp"
#include <memory>

#include "Utils/IOTensor.hpp"
#include "PAL/DynamicLoading.hpp"
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
        if (sg_backendHandle) {
          pal::dynamicloading::dlClose(sg_backendHandle);
        }
        if (sg_modelHandle) {
          pal::dynamicloading::dlClose(sg_modelHandle);
        }
        QNN_INFO("Free handle");
    }

    // Init QNN Backend context
    void init();
    int32_t r_init(); // TODO: Config

    void release();
    int32_t r_release();

    // void alloc(void **ptr, size_t size,size_t alignment) {
    //     mem_manager_->alloc(ptr, size, alignment);
    // }

    // void free(void *ptr) {
    //     mem_manager_->free(ptr);
    // }

    
    /**
     * @brief create execution for op with input and output tensors.
     * @param inputs    input tensors.
     * @param outputs   output tensors.
     * @param op        given op.
     * @return created execution if op is supported, nullptr otherwise.
     */
    // virtual Op* OpCreate(const vector<shared_ptr<Tensor>>& inputs, const vector<shared_ptr<Tensor>>& outputs,
    //                             OpParam op_param) = 0;
    Op *opCreate(const OpParam &op_param, string name="") {

    }
    void registerOps() {

    }
    // virtual void* OpCreater(OpParam op_param);


    // @brief Print a message to STDERR then return a nonzero
    //  exit status.
    int32_t reportError(const std::string &err);

// private:
    //
    shared_ptr<MemoryManager> mem_manager_;
    // unordered_map<OpType, Op*(*)(Backend*)> op_map_;

    // --------- temp dev functions to test QNNBackend
    ErrorCode graphAddNode(Op op);
    ErrorCode graphFinilize();
    ErrorCode graphExecute();
    // ---------
    
    StatusCode initialize();

    StatusCode initializeBackend();

    StatusCode createContext();

    StatusCode composeGraphs();

    StatusCode finalizeGraphs();

    StatusCode executeGraphs();

    StatusCode registerOpPackages();

    StatusCode createFromBinary();

    StatusCode saveBinary();

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


    void QnnBackendInitialize(sample_app::QnnFunctionPointers qnnFunctionPointers,
               std::string inputListPaths,
               std::string opPackagePaths,
               void *backendHandle,
               std::string outputPath                  = s_defaultOutputPath,
               bool debug                              = false,
               iotensor::OutputDataType outputDataType = iotensor::OutputDataType::FLOAT_ONLY,
               iotensor::InputDataType inputDataType   = iotensor::InputDataType::FLOAT,
               sample_app::ProfilingLevel profilingLevel           = sample_app::ProfilingLevel::OFF,
               bool dumpOutputs                        = false,
               std::string cachedBinaryPath            = "",
               std::string saveBinaryName              = "");

    static const std::string s_defaultOutputPath;


    sample_app::QnnFunctionPointers m_qnnFunctionPointers;
    std::vector<std::string> m_inputListPaths;
    std::vector<std::vector<std::queue<std::string>>> m_inputFileLists;
    std::vector<std::string> m_opPackagePaths;
    std::string m_outputPath;
    std::string m_saveBinaryName;
    std::string m_cachedBinaryPath;
    QnnBackend_Config_t **m_backendConfig = nullptr;
    Qnn_ContextHandle_t m_context         = nullptr;
    QnnContext_Config_t **m_contextConfig = nullptr;
    bool m_debug;
    iotensor::OutputDataType m_outputDataType;
    iotensor::InputDataType m_inputDataType;
    sample_app::ProfilingLevel m_profilingLevel;
    bool m_dumpOutputs;
    qnn_wrapper_api::GraphInfo_t **m_graphsInfo;
    uint32_t m_graphsCount;
    void *m_backendLibraryHandle;
    iotensor::IOTensor m_ioTensor;
    bool m_isBackendInitialized;
    bool m_isContextCreated;
    Qnn_ProfileHandle_t m_profileBackendHandle              = nullptr;
    qnn_wrapper_api::GraphConfigInfo_t **m_graphConfigsInfo = nullptr;
    uint32_t m_graphConfigsInfoCount;
    Qnn_LogHandle_t m_logHandle         = nullptr;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_DeviceHandle_t m_deviceHandle   = nullptr;


    void* sg_backendHandle = nullptr;
    void* sg_modelHandle = nullptr;

};


} // namespace mllm

#endif // MLLM_QNNBACKEND_H