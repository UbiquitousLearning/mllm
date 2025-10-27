#ifndef MLLM_QNNBACKEND_H
#define MLLM_QNNBACKEND_H

#include "Backend.hpp"
#include "Op.hpp"
#include "OpDefined.hpp"
#include "ParamLoader.hpp"
#include "QNNUtils.hpp"
#include "QNNModel.hpp"
#include "QnnTypes.h"
#include "HTP/QnnHtpDevice.h"
#include "System/QnnSystemInterface.h"
#include "Types.hpp"
#include "MemoryManager.hpp"
#include <memory>

namespace mllm {
class Module;
class Layer;
class Op;
class Tensor;
class Backend;

enum class ProfilingLevel { OFF,
                            BASIC,
                            DETAILED,
                            INVALID };
class QNNPerf {
public:
    static std::unique_ptr<QNNPerf> create(const QNN_INTERFACE_VER_TYPE *qnnInterface) {
        return std::unique_ptr<QNNPerf>(new QNNPerf(qnnInterface));
    }
    QNNPerf(const QNN_INTERFACE_VER_TYPE *qnnInterface);
    ~QNNPerf();
    void setRpcLatencyAndPolling();
    void setPowerConfigBurst();
    void setPowerConfigBalanced();

private:
    const QNN_INTERFACE_VER_TYPE *mQnnInterface = nullptr;
    QnnHtpDevice_PerfInfrastructure_t mPerfInfra{};
    uint32_t mPowerConfigId;
    QnnHtpPerfInfrastructure_PowerConfig_t mPowerConfigBurst{};
    QnnHtpPerfInfrastructure_PowerConfig_t mPowerConfigBalanced{};
};

class QNNRuntime {
    friend class QNNBackend;

public:
    ~QNNRuntime();

    static std::unique_ptr<QNNRuntime> create(ProfilingLevel profilingLevel = ProfilingLevel::OFF, QnnLog_Level_t qnnLogLevel = QNN_LOG_LEVEL_INFO) {
        return std::unique_ptr<QNNRuntime>(initRuntime(profilingLevel, qnnLogLevel));
    }

    bool createContext(Qnn_ContextHandle_t &context, QnnContext_Config_t **contextConfig = nullptr);
    bool retrieveContext(Qnn_ContextHandle_t &context,
                         std::vector<GraphInfo_t *> &graphsInfo,
                         QnnContext_Config_t **contextConfig = nullptr);

private:
    QNN_INTERFACE_VER_TYPE qnnInterface;
    QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface;

    Qnn_LogHandle_t logHandle = nullptr;
    Qnn_BackendHandle_t backendHandle = nullptr;
    Qnn_DeviceHandle_t deviceHandle = nullptr;
    Qnn_ProfileHandle_t profileHandle = nullptr;

    QNNRuntime(QNN_INTERFACE_VER_TYPE qnnInterface,
               QNN_SYSTEM_INTERFACE_VER_TYPE qnnSystemInterface,
               Qnn_LogHandle_t qnnLogHandle,
               Qnn_BackendHandle_t qnnBackendHandle,
               Qnn_DeviceHandle_t qnnDeviceHandle,
               Qnn_ProfileHandle_t qnnProfileHandle = nullptr) :
        qnnInterface(qnnInterface),
        qnnSystemInterface(qnnSystemInterface), logHandle(qnnLogHandle), backendHandle(qnnBackendHandle), deviceHandle(qnnDeviceHandle), profileHandle(qnnProfileHandle) {
    }

    std::string getBackendBuildId(QNN_INTERFACE_VER_TYPE &qnnInterface) {
        char *backendBuildId{nullptr};
        if (QNN_SUCCESS != qnnInterface.backendGetBuildId((const char **)&backendBuildId)) {
            MLLM_LOG_ERROR_LEGACY("Unable to get build Id from the backend.");
        }
        return (backendBuildId == nullptr ? std::string("") : std::string(backendBuildId));
    }

    static QNNRuntime *initRuntime(ProfilingLevel profilingLevel, QnnLog_Level_t qnnLogLevel);
};

class QNNBackend : public Backend {
public:
    QNNBackend(shared_ptr<MemoryManager> mm);
    ~QNNBackend();

    Op *opCreate(const OpParam &op_param, string name = "", int threadCount = 4) override {
        OpType optype = OpType(op_param.find("type")->second);
        auto iter = map_creator_.find(optype);
        if (iter == map_creator_.end()) {
            std::cout << "NPU Op Don't support type : " << optype << ", name" << name << std::endl;
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

    void graphAddNode(string name, string nodeType,
                      std::vector<string> inputTensorNames, std::vector<Qnn_Tensor_t> outputTensors,
                      std::vector<Qnn_Param_t> params,
                      string packageName);

    void modelAddTensor(std::string nodeName, Qnn_Tensor_t tensor);

    virtual void onSetUpStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) override;
    virtual void onSetUpEnd(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName) override;
    virtual void onExecuteStart(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs, string graphName = "") override;
    virtual void onExecuteEnd(std::vector<shared_ptr<Tensor>> &outputs, const string &graph_name) override{};

    // std::vector<Tensor> runFunc(
    //     std::vector<std::string> out_names,
    //     TensorFuncType type,
    //     std::vector<float> float_args,
    //     std::vector<Tensor> input_tensors,
    //     bool in_place) override;
    std::vector<Tensor> runLayer(Layer *layer, std::vector<Tensor> inputs, int N) override;
    std::vector<Tensor> runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) override;
    std::vector<Tensor> runOp(Op *op, std::vector<Tensor> input, std::vector<std::string> out_names, bool in_place) override;

    void pushInputBuffers(uint8_t *ptr) {
        currentInputBuffers->push_back(ptr);
    }
    void pushOutputBuffers(uint8_t *ptr) {
        currentOutputBuffers->push_back(ptr);
    }

    void saveQNNContext();

private:
    bool graphFinilize();

    void registerOps() override;
    void registerFuncs() override{};

    void extractBackendProfilingInfo(Qnn_ProfileHandle_t profileHandle);

    void extractProfilingSubEvents(QnnProfile_EventId_t profileEventId);

    void extractProfilingEvent(QnnProfile_EventId_t profileEventId);

    std::map<std::string, std::vector<uint8_t *>> inputBufferMap, outputBufferMap;
    // still use this, as in Express frontend, mllm inputs and outputs num may not match
    std::vector<uint8_t *> *currentInputBuffers, *currentOutputBuffers;

    std::map<OpType, QNNBackend::Creator *> map_creator_;

    std::map<std::string, int> qnnModelIndexMap_;
    std::vector<QNNModel> qnnModels_;
    int qnnModelIndex_;

    Qnn_ContextHandle_t m_context = nullptr;
    bool m_debug;

    ProfilingLevel m_profilingLevel;

    std::vector<GraphInfo_t *> graphsInfo_;

    bool isFromCache = false;

    std::unique_ptr<QNNRuntime> mRuntime;
    std::unique_ptr<QNNPerf> mPerf;
};

} // namespace mllm

#endif // MLLM_QNNBACKEND_H