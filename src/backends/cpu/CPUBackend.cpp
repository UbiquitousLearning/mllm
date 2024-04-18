#include "CPUBackend.hpp"
#include <math.h>
#include <memory>
#include "Backend.hpp"
#include "CPUMatmulINT8.hpp"
#include "CPUPoEmbedding.hpp"
#include "CPUSplitInput.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "CPUView.hpp"
#include "CPUAdd.hpp"
#include "CPUCausalMask.hpp"
#include "CPUMatmul.hpp"
#include "CPURMSNorm.hpp"
#include "CPURoPE.hpp"
#include "CPUScale.hpp"
#include "CPUSiLU.hpp"
#include "CPUSoftMax.hpp"
#include "CPULinear.hpp"
#include "CPUEmbedding.hpp"
#include "CPUMul.hpp"
#include "CPUKVCache.hpp"
#include "CPUReLU.hpp"
#include "CPUReLU2.hpp"
#include "CPUGELU.hpp"
#include "CPUSplit.hpp"
#include "CPULayerNorm.hpp"
#include "CPUGather.hpp"
#include "CPUConvolution2D.hpp"
#include "CPUAvgPool2D.hpp"
#include "CPUMaxPool2D.hpp"
#include "CPUConvolution3D.hpp"
#include "CPUParameter.hpp"
#include "CPUCat.hpp"
#include "CPUSubDim.hpp"
#include "CPUQuickGELU.hpp"
#include "CPUDivision.hpp"
#include "CPUNorm.hpp"
#include "CPUShape.hpp"
#include "CPUTranspose.hpp"
#include "CPUMean.hpp"
#include "CPURange.hpp"
#include "CPUWhere.hpp"
#include "CPUReplace.hpp"
#include "CPUQuantize.hpp"
#include "CPUMergeOutput.hpp"



namespace mllm {
class CPUBackendCreator : public BackendCreator {
    shared_ptr<Backend> create(BackendConfig config) {
        shared_ptr<MemoryManager> mm = nullptr;
        switch (config.memory) {
        case BackendConfig::Memory_High:
            mm = std::make_shared<SystemMemoryManager>();
            break;
        default:
            mm = std::make_shared<SystemMemoryManager>();
            break;
        }
        return std::make_shared<CPUBackend>(mm);
    };
};

void registerCPUBackendCreator() {
    InsertBackendCreatorMap(MLLM_CPU, std::make_shared<CPUBackendCreator>());
}

CPUBackend::CPUBackend(shared_ptr<MemoryManager>& mm) :
    Backend(mm) {
    registerOps();
}

Op *CPUBackend::opCreate(const OpParam &op_param, string name, int threadCount) {
    OpType optype = OpType(op_param.find("type")->second);
    auto iter = map_creator_.find(optype);
    if (iter == map_creator_.end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    Op *exe = iter->second->create(op_param, this, name, threadCount);
    return exe;
}
void CPUBackend::registerOps() {
    addCreator(PARAMETER, (CPUBackend::Creator *)(new CPUParameterCreator()));
    addCreator(ADD, (CPUBackend::Creator *)(new CPUAddCreator()));
    addCreator(CAUSALMASK, (CPUBackend::Creator *)(new CPUCausalMaskCreator()));
    addCreator(MATMUL, (CPUBackend::Creator *)(new CPUMatmulCreator()));
    addCreator(MATMULINT8, (CPUBackend::Creator *)(new CPUMatmulINT8Creator()));
    addCreator(RMSNORM, (CPUBackend::Creator *)(new CPURMSNormCreator()));
    addCreator(ROPE, (CPUBackend::Creator *)(new CPURoPECreator()));
    addCreator(SCALE, (CPUBackend::Creator *)(new CPUScaleCreator()));
    addCreator(SILU, (CPUBackend::Creator *)(new CPUSiLUCreator()));
    addCreator(SOFTMAX, (CPUBackend::Creator *)(new CPUSoftMaxCreator()));
    addCreator(LINEAR, (CPUBackend::Creator *)(new CPULinearCreator()));
    addCreator(EMBEDDING, (CPUBackend::Creator *)(new CPUEmbeddingCreator()));
    addCreator(MUL, (CPUBackend::Creator *)(new CPUMulCreator()));
    addCreator(VIEW, (CPUBackend::Creator *)(new CPUViewCreator()));
    addCreator(KVCACHE, (CPUBackend::Creator *)(new CPUKVCacheCreator()));
    addCreator(RELU, (CPUBackend::Creator *)(new CPUReLUCreator()));
    addCreator(RELU2, (CPUBackend::Creator *)(new CPUReLU2Creator()));
    addCreator(GELU, (CPUBackend::Creator *)(new CPUGELUCreator()));
    addCreator(QUICKGLUE, (CPUBackend::Creator *)(new CPUQuickGELUCreator()));
    addCreator(LAYERNORM, (CPUBackend::Creator *)(new CPULayerNormCreator()));
    addCreator(SPLIT, (CPUBackend::Creator *)(new CPUSplitCreator()));
    addCreator(GATHER, (CPUBackend::Creator *)(new CPUGatherCreator()));
    addCreator(CONVOLUTION2D, (CPUBackend::Creator *)(new CPUConvolution2DCreator()));
    addCreator(AVGPOOL2D, (CPUBackend::Creator *)(new CPUAvgPoolCreator()));
    addCreator(MAXPOOL2D, (CPUBackend::Creator *)(new CPUMaxPoolCreator()));
    addCreator(CONVOLUTION3D, (CPUBackend::Creator *)(new CPUConvolution3DCreator()));
    addCreator(CAT, (CPUBackend::Creator *)(new CPUCatCreator()));
    addCreator(TRANSPOSE, (CPUBackend::Creator *)(new CPUTransposeCreator()));
    addCreator(SUBDIM, (CPUBackend::Creator *)(new CPUSubDimCreator()));
    addCreator(DIVISION, (CPUBackend::Creator *)(new CPUDivisionCreator()));
    addCreator(NORM, (CPUBackend::Creator *)(new CPUNormCreator()));
    addCreator(SHAPE, (CPUBackend::Creator *)(new CPUShapeCreator()));
    addCreator(MEAN, (CPUBackend::Creator *)(new CPUMeanCreator()));
    addCreator(RANGE, (CPUBackend::Creator *)(new CPURangeCreator()));
    addCreator(WHERE, (CPUBackend::Creator *)(new CPUWhereCreator()));
    addCreator(REPLACE, (CPUBackend::Creator *)(new CPUReplaceCreator()));
    addCreator(POSITIOANL_EMBEDDING, (CPUBackend::Creator *)(new CPUPoEmbeddingCreator()));
    addCreator(SPLITINPUT, (CPUBackend::Creator *)(new CPUSplitInputCreator()));
    addCreator(QUANTIZE, (CPUBackend::Creator *)(new CPUQuantizeCreator()));
    addCreator(MERGEOUTPUT, (CPUBackend::Creator *)(new CPUMergeOutputCreator()));
}

} // namespace mllm
