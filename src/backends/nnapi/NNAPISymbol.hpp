#ifndef MLLM_NNAPISYMBOLS_H
#define MLLM_NNAPISYMBOLS_H

#include "NNAPIDefine.hpp"

namespace mllm {
// typedef the function in nnapi will be used
typedef int(ANeuralNetworksModel_getSupportedOperationsForDevices)(const ANeuralNetworksModel *model, const ANeuralNetworksDevice *const *devices, uint32_t numDevices, bool *supportedOps);
typedef int(ANeuralNetworks_getDeviceCount)(uint32_t *numDevices);
typedef int(ANeuralNetworks_getDevice)(uint32_t devIndex, ANeuralNetworksDevice **device);
typedef int(ANeuralNetworksDevice_getName)(const ANeuralNetworksDevice *device, const char **name);
typedef int(ANeuralNetworksDevice_getType)(const ANeuralNetworksDevice *device, int32_t *type);
typedef int(ANeuralNetworksCompilation_createForDevices)(ANeuralNetworksModel *model, const ANeuralNetworksDevice *const *devices, uint32_t numDevices, ANeuralNetworksCompilation **compilation);
typedef int(ANeuralNetworksExecution_compute)(ANeuralNetworksExecution *execution);
typedef int(ANeuralNetworksBurst_create)(ANeuralNetworksCompilation *compilation, ANeuralNetworksBurst **burst);
typedef void(ANeuralNetworksBurst_free)(ANeuralNetworksBurst *burst);
typedef int(ANeuralNetworksExecution_burstCompute)(ANeuralNetworksExecution *execution, ANeuralNetworksBurst *burst);
typedef int(ANeuralNetworksModel_create)(ANeuralNetworksModel **model);
typedef void(ANeuralNetworksModel_free)(ANeuralNetworksModel *model);
typedef int(ANeuralNetworksModel_finish)(ANeuralNetworksModel *model);
typedef int(ANeuralNetworksModel_addOperand)(ANeuralNetworksModel *model, const ANeuralNetworksOperandType *type);
typedef int(ANeuralNetworksModel_setOperandValue)(ANeuralNetworksModel *model, int32_t index, const void *buffer, size_t length);
typedef int(ANeuralNetworksModel_setOperandSymmPerChannelQuantParams)(ANeuralNetworksModel *model, int32_t index, const ANeuralNetworksSymmPerChannelQuantParams *channelQuant);
typedef int(ANeuralNetworksModel_addOperation)(ANeuralNetworksModel *model, ANeuralNetworksOperationType type, uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
typedef int(ANeuralNetworksModel_identifyInputsAndOutputs)(ANeuralNetworksModel *model, uint32_t inputCount, const uint32_t *inputs, uint32_t outputCount, const uint32_t *outputs);
typedef int(ANeuralNetworksCompilation_create)(ANeuralNetworksModel *model, ANeuralNetworksCompilation **compilation);
typedef void(ANeuralNetworksCompilation_free)(ANeuralNetworksCompilation *compilation);
typedef int(ANeuralNetworksCompilation_setPreference)(ANeuralNetworksCompilation *compilation, int32_t preference);
typedef int(ANeuralNetworksCompilation_finish)(ANeuralNetworksCompilation *compilation);
typedef int(ANeuralNetworksExecution_create)(ANeuralNetworksCompilation *compilation, ANeuralNetworksExecution **execution);
typedef void(ANeuralNetworksExecution_free)(ANeuralNetworksExecution *execution);
typedef int(ANeuralNetworksExecution_setInput)(ANeuralNetworksExecution *execution, int32_t index, const ANeuralNetworksOperandType *type, const void *buffer, size_t length);
typedef int(ANeuralNetworksExecution_setInputFromMemory)(ANeuralNetworksExecution *execution, int32_t index, const ANeuralNetworksOperandType *type, const ANeuralNetworksMemory *memory, size_t offset, size_t length);
typedef int(ANeuralNetworksExecution_setOutput)(ANeuralNetworksExecution *execution, int32_t index, const ANeuralNetworksOperandType *type, void *buffer, size_t length);
typedef int(ANeuralNetworksExecution_setOutputFromMemory)(ANeuralNetworksExecution *execution, int32_t index, const ANeuralNetworksOperandType *type, const ANeuralNetworksMemory *memory, size_t offset, size_t length);
typedef int(ANeuralNetworksExecution_startCompute)(ANeuralNetworksExecution *execution, ANeuralNetworksEvent **event);
typedef int(ANeuralNetworksEvent_wait)(ANeuralNetworksEvent *event);
typedef void(ANeuralNetworksEvent_free)(ANeuralNetworksEvent *event);
typedef int(ANeuralNetworksDevice_getVersion)(const ANeuralNetworksDevice *device, const char **version);
typedef int(ANeuralNetworksMemory_createFromAHardwareBuffer)(const AHardwareBuffer *ahwb, ANeuralNetworksMemory **memory);
typedef int(ANeuralNetworksMemory_createFromFd)(size_t size, int protect, int fd, size_t offset, ANeuralNetworksMemory **memory);
typedef void(ANeuralNetworksMemory_free)(ANeuralNetworksMemory *memory);
typedef void(ANeuralNetworksExecution_setMeasureTiming)(ANeuralNetworksExecution *execution, bool measure);
typedef void(ANeuralNetworksExecution_getDuration)(const ANeuralNetworksExecution *execution, int32_t durationCode, uint64_t *duration);

// symbols
bool loadNNAPISymbol();
extern ANeuralNetworksModel_getSupportedOperationsForDevices *ANeuralNetworksModel_getSupportedOperationsForDevices_29;
extern ANeuralNetworks_getDeviceCount *ANeuralNetworks_getDeviceCount_29;
extern ANeuralNetworks_getDevice *ANeuralNetworks_getDevice_29;
extern ANeuralNetworksDevice_getName *ANeuralNetworksDevice_getName_29;
extern ANeuralNetworksDevice_getType *ANeuralNetworksDevice_getType_29;
extern ANeuralNetworksCompilation_createForDevices *ANeuralNetworksCompilation_createForDevices_29;
extern ANeuralNetworksExecution_compute *ANeuralNetworksExecution_compute_29;
extern ANeuralNetworksBurst_create *ANeuralNetworksBurst_create_29;
extern ANeuralNetworksBurst_free *ANeuralNetworksBurst_free_29;
extern ANeuralNetworksExecution_burstCompute *ANeuralNetworksExecution_burstCompute_29;
extern ANeuralNetworksModel_create *ANeuralNetworksModel_create_27;
extern ANeuralNetworksModel_free *ANeuralNetworksModel_free_27;
extern ANeuralNetworksModel_finish *ANeuralNetworksModel_finish_27;
extern ANeuralNetworksModel_addOperand *ANeuralNetworksModel_addOperand_27;
extern ANeuralNetworksModel_setOperandValue *ANeuralNetworksModel_setOperandValue_27;
extern ANeuralNetworksModel_setOperandSymmPerChannelQuantParams *ANeuralNetworksModel_setOperandSymmPerChannelQuantParams_29;
extern ANeuralNetworksModel_addOperation *ANeuralNetworksModel_addOperation_27;
extern ANeuralNetworksModel_identifyInputsAndOutputs *ANeuralNetworksModel_identifyInputsAndOutputs_27;
extern ANeuralNetworksCompilation_create *ANeuralNetworksCompilation_create_27;
extern ANeuralNetworksCompilation_free *ANeuralNetworksCompilation_free_27;
extern ANeuralNetworksCompilation_setPreference *ANeuralNetworksCompilation_setPreference_27;
extern ANeuralNetworksCompilation_finish *ANeuralNetworksCompilation_finish_27;
extern ANeuralNetworksExecution_create *ANeuralNetworksExecution_create_27;
extern ANeuralNetworksExecution_free *ANeuralNetworksExecution_free_27;
extern ANeuralNetworksExecution_setInput *ANeuralNetworksExecution_setInput_27;
extern ANeuralNetworksExecution_setInputFromMemory *ANeuralNetworksExecution_setInputFromMemory_27;
extern ANeuralNetworksExecution_setOutput *ANeuralNetworksExecution_setOutput_27;
extern ANeuralNetworksExecution_setOutputFromMemory *ANeuralNetworksExecution_setOutputFromMemory_27;
extern ANeuralNetworksExecution_startCompute *ANeuralNetworksExecution_startCompute_27;
extern ANeuralNetworksEvent_wait *ANeuralNetworksEvent_wait_27;
extern ANeuralNetworksEvent_free *ANeuralNetworksEvent_free_27;
extern ANeuralNetworksDevice_getVersion *ANeuralNetworksDevice_getVersion_29;
extern ANeuralNetworksMemory_createFromAHardwareBuffer *ANeuralNetworksMemory_createFromAHardwareBuffer_29;
extern ANeuralNetworksMemory_createFromFd *ANeuralNetworksMemory_createFromFd_27;
extern ANeuralNetworksMemory_free *ANeuralNetworksMemory_free_27;
extern ANeuralNetworksExecution_setMeasureTiming *ANeuralNetworksExecution_setMeasureTiming_29;
extern ANeuralNetworksExecution_getDuration *ANeuralNetworksExecution_getDuration_29;
} // namespace mllm
#endif // MLLM_NNAPISYMBOLS_H