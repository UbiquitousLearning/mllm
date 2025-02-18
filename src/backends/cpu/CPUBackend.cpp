#include "CPUBackend.hpp"
#include <iostream>
#include <math.h>
#include <memory>
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include "memory/SystemMemoryManager.hpp"

#include "op/CPUHeadLinear.hpp"
#include "op/CPULinearInt8.hpp"
#include "op/CPUNTKRoPE.hpp"
#include "op/CPUPoEmbedding.hpp"
#include "op/CPUSplitInput.hpp"
#include "op/CPUView.hpp"
#include "op/CPUAdd.hpp"
#include "op/CPUCausalMask.hpp"
#include "op/CPUSlidingWindowMask.hpp"
#include "op/CPUMatmul.hpp"
#include "op/CPURMSNorm.hpp"
#include "op/CPURoPE.hpp"
#include "op/CPUScale.hpp"
#include "op/CPUSiLU.hpp"
#include "op/CPUSoftMax.hpp"
#include "op/CPULinear.hpp"
#include "op/CPUEmbedding.hpp"
#include "op/CPUMul.hpp"
#include "op/CPUKVCache.hpp"
#include "op/CPUReLU.hpp"
#include "op/CPUReLU2.hpp"
#include "op/CPUGELU.hpp"
#include "op/CPUSplit.hpp"
#include "op/CPULayerNorm.hpp"
#include "op/CPUGather.hpp"
#include "op/CPUConvolution2D.hpp"
#include "op/CPUAvgPool2D.hpp"
#include "op/CPUMaxPool2D.hpp"
#include "op/CPUConvolution3D.hpp"
#include "op/CPUVisionRoPE.hpp"
#include "op/CPUMultimodalRoPE.hpp"
#include "op/CPUParameter.hpp"
#include "op/CPUCat.hpp"
#include "op/CPUSubDim.hpp"
#include "op/CPUQuickGELU.hpp"
#include "op/CPUDivision.hpp"
#include "op/CPUNorm.hpp"
#include "op/CPUShape.hpp"
#include "op/CPUTranspose.hpp"
#include "op/CPUMean.hpp"
#include "op/CPURange.hpp"
#include "op/CPUWhere.hpp"
#include "op/CPUReplace.hpp"
#include "op/CPUPredictor.hpp"
#include "op/CPUSparseIdLinear.hpp"
#include "op/CPUSparseLinear.hpp"
#include "op/CPUElasticLinear.hpp"
#include "op/CPUQuantize.hpp"
#include "op/CPUMergeOutput.hpp"
#include "op/CPULinearINT8Shadow.hpp"
#include "op/CPUIRoPE.hpp"
#include "op/CPUPosition.hpp"

#include "op/CPUKVCacheNPU.hpp"
#include "op/CPUKVCacheXp.hpp"

#include "function/CPUBinaryFunc.hpp"
#include "function/CPUCatFunc.hpp"
#include "function/CPUClipFunc.hpp"
#include "function/CPUExpandFunc.hpp"
#include "function/CPUFlattenFunc.hpp"
#include "function/CPUMatmulFunc.hpp"
#include "function/CPUMeanFunc.hpp"
#include "function/CPUNormFunc.hpp"
#include "function/CPURangeFunc.hpp"
#include "function/CPUSplitFunc.hpp"
#include "function/CPUSumFunc.hpp"
#include "function/CPUTopkFunc.hpp"
#include "function/CPUTransposeFunc.hpp"
#include "function/CPUViewFunc.hpp"
#include "function/CPUWhereFunc.hpp"
#include "function/CPUIndexPutFunc.hpp"
#include "function/CPUArgSortFunc.hpp"
#include "function/CPUBinCountFunc.hpp"
#include "function/CPURepeatFunc.hpp"
#include "function/CPULikeFunc.hpp"
#include "function/CPUScatterReduceFunc.hpp"
#include "function/CPUApplyVisionRoPE.hpp"

#include "function/CPUFuyuGatherEmbdFunc.hpp"
#include "function/CPUPhi3VhdmergeFunc.hpp"

namespace mllm {
class CPUBackendCreator : public BackendCreator {
    Backend *create(BackendConfig config) {
        shared_ptr<MemoryManager> mm = nullptr;
        switch (config.memory) {
        case BackendConfig::Memory_High:
            mm = std::make_shared<SystemMemoryManager>();
            break;
        default:
            mm = std::make_shared<SystemMemoryManager>();
            break;
        }
        return new CPUBackend(mm);
    };
};

void registerCPUBackendCreator() {
    InsertBackendCreatorMap(MLLM_CPU, std::make_shared<CPUBackendCreator>());
}

CPUBackend::CPUBackend(shared_ptr<MemoryManager> &mm) :
    Backend(mm) {
    type_ = BackendType::MLLM_CPU;
    registerOps();
    registerFuncs();
}

Op *CPUBackend::opCreate(const OpParam &op_param, string name, int threadCount) {
    OpType optype = OpType(op_param.find("type")->second);
    auto iter = map_creator_.find(optype);
    if (iter == map_creator_.end()) {
        std::cout << "CPU Op Don't support type : " << name << std::endl;
        return nullptr;
    }
    Op *exe = iter->second->create(op_param, this, name, cpu_threads);
    return exe;
}
void CPUBackend::registerOps() {
    addCreator(PARAMETER, (CPUBackend::Creator *)(new CPUParameterCreator()));
    addCreator(ADD, (CPUBackend::Creator *)(new CPUAddCreator()));
    addCreator(CAUSALMASK, (CPUBackend::Creator *)(new CPUCausalMaskCreator()));
    addCreator(SLIDINGWINDOWMASK, (CPUBackend::Creator *)(new CPUSlidingWindowMaskCreator()));
    addCreator(MATMUL, (CPUBackend::Creator *)(new CPUMatmulCreator()));
    addCreator(RMSNORM, (CPUBackend::Creator *)(new CPURMSNormCreator()));
    addCreator(ROPE, (CPUBackend::Creator *)(new CPURoPECreator()));
    addCreator(SCALE, (CPUBackend::Creator *)(new CPUScaleCreator()));
    addCreator(SILU, (CPUBackend::Creator *)(new CPUSiLUCreator()));
    addCreator(SOFTMAX, (CPUBackend::Creator *)(new CPUSoftMaxCreator()));
    addCreator(LINEAR, (CPUBackend::Creator *)(new CPULinearCreator()));
    addCreator(LINEARINT8, (CPUBackend::Creator *)(new CPULinearInt8Creator()));
    addCreator(EMBEDDING, (CPUBackend::Creator *)(new CPUEmbeddingCreator()));
    addCreator(MUL, (CPUBackend::Creator *)(new CPUMulCreator()));
    addCreator(VIEW, (CPUBackend::Creator *)(new CPUViewCreator()));
    addCreator(KVCACHE, (CPUBackend::Creator *)(new CPUKVCacheCreator()));
    addCreator(KVCACHENPU, (CPUBackend::Creator *)(new CPUKVCacheNPUCreator()));
    addCreator(RELU, (CPUBackend::Creator *)(new CPUReLUCreator()));
    addCreator(RELU2, (CPUBackend::Creator *)(new CPUReLU2Creator()));
    addCreator(OP_GELU, (CPUBackend::Creator *)(new CPUGELUCreator()));
    addCreator(QUICKGLUE, (CPUBackend::Creator *)(new CPUQuickGELUCreator()));
    addCreator(LAYERNORM, (CPUBackend::Creator *)(new CPULayerNormCreator()));
    addCreator(SPLIT, (CPUBackend::Creator *)(new CPUSplitCreator()));
    addCreator(GATHER, (CPUBackend::Creator *)(new CPUGatherCreator()));
    addCreator(CONVOLUTION2D, (CPUBackend::Creator *)(new CPUConvolution2DCreator()));
    addCreator(AVGPOOL2D, (CPUBackend::Creator *)(new CPUAvgPoolCreator()));
    addCreator(MAXPOOL2D, (CPUBackend::Creator *)(new CPUMaxPoolCreator()));
    addCreator(CONVOLUTION3D, (CPUBackend::Creator *)(new CPUConvolution3DCreator()));
    addCreator(VISIONROPE, (CPUBackend::Creator *)(new CPUVisionRoPECreator()));
    addCreator(MULTIMODALROPE, (CPUBackend::Creator *)(new CPUMultimodalRoPECreator()));
    // addCreator(CAT, (CPUBackend::Creator *)(new CPUCatCreator()));
    addCreator(TRANSPOSE, (CPUBackend::Creator *)(new CPUTransposeCreator()));
    addCreator(SUBDIM, (CPUBackend::Creator *)(new CPUSubDimCreator()));
    addCreator(DIVISION, (CPUBackend::Creator *)(new CPUDivisionCreator()));
    addCreator(NORM, (CPUBackend::Creator *)(new CPUNormCreator()));
    addCreator(SHAPE, (CPUBackend::Creator *)(new CPUShapeCreator()));
    addCreator(MEAN, (CPUBackend::Creator *)(new CPUMeanCreator()));
    addCreator(RANGE, (CPUBackend::Creator *)(new CPURangeCreator()));
    addCreator(WHERE, (CPUBackend::Creator *)(new CPUWhereCreator()));
    addCreator(REPLACE, (CPUBackend::Creator *)(new CPUReplaceCreator()));
    addCreator(PREDICTOR, (CPUBackend::Creator *)(new CPUPredictorCreator()));
    addCreator(SPARSELINEAR, (CPUBackend::Creator *)(new CPUSparseLinearCreator()));
    addCreator(SPARSEIDLINEAR, (CPUBackend::Creator *)(new CPUSparseIdLinearCreator()));
    addCreator(ELASTICLINEAR, (CPUBackend::Creator *)(new CPUElasticLinearCreator()));
    addCreator(POSITION, (CPUBackend::Creator *)(new CPUPositionCreator()));
    addCreator(QUANTIZE, (CPUBackend::Creator *)(new CPUQuantizeCreator()));
    addCreator(MERGEOUTPUT, (CPUBackend::Creator *)(new CPUMergeOutputCreator()));
    addCreator(SPLITINPUT, (CPUBackend::Creator *)(new CPUSplitInputCreator()));
    addCreator(LINEARINT8SHADOW, (CPUBackend::Creator *)(new CPULinearINT8ShadowCreator()));
    addCreator(IROPE, (CPUBackend::Creator *)(new CPUIRoPECreator()));
    addCreator(XP_KVCACHE, (CPUBackend::Creator *)(new CPUKVCacheXpCreator()));
    addCreator(NTKROPE, (CPUBackend::Creator *)(new CPUNTKRoPECreator()));
    addCreator(HEADLINEAR, (CPUBackend::Creator *)(new CPUHeadLinearCreator()));
}
TensorFunction *CPUBackend::funcCreate(const TensorFuncType type) {
    auto iter = map_function_.find(type);
    if (iter == map_function_.end()) {
        std::cout << "CPU funcCreate Don't support type : " << type << std::endl;
        return nullptr;
    }
    return iter->second;
}

void CPUBackend::registerFuncs() {
    map_function_[TensorFuncType::FUNC_ADD] = new CPUaddFunction();
    map_function_[TensorFuncType::FUNC_SUB] = new CPUsubFunction();
    map_function_[TensorFuncType::FUNC_MUL] = new CPUmulFunction();
    map_function_[TensorFuncType::FUNC_DIV] = new CPUdivFunction();
    map_function_[TensorFuncType::FUNC_DIVINT] = new CPUdivintFunction();
    map_function_[TensorFuncType::FUNC_TTADD] = new CPUaddTwoFunction();
    map_function_[TensorFuncType::FUNC_TTSUB] = new CPUsubTwoFunction();
    map_function_[TensorFuncType::FUNC_TTMUL] = new CPUmulTwoFunction();
    map_function_[TensorFuncType::FUNC_TTDIV] = new CPUdivTwoFunction();
    map_function_[TensorFuncType::FUNC_MM] = new CPUmmFunction();
    map_function_[TensorFuncType::FUNC_NORM] = new CPUnormFunction();
    map_function_[TensorFuncType::FUNC_MEAN] = new CPUmeanFunction();
    map_function_[TensorFuncType::FUNC_CAT] = new CPUcatFunction();
    map_function_[TensorFuncType::FUNC_VIEW] = new CPUviewFunction();
    map_function_[TensorFuncType::FUNC_TRANPOSE] = new CPUtransposeFunction();
    map_function_[TensorFuncType::FUNC_FLATTEN] = new CPUflattenFunction();
    map_function_[TensorFuncType::FUNC_CLIP] = new CPUclipFunction();
    map_function_[TensorFuncType::FUNC_CLIPAXIS] = new CPUclipaxisFunction();
    map_function_[TensorFuncType::FUNC_CLIPTENSOR] = new CPUcliptensorFunction();
    map_function_[TensorFuncType::FUNC_RANGE] = new CPURangeFunction();
    map_function_[TensorFuncType::FUNC_WHERE] = new CPUwhereFunction();
    map_function_[TensorFuncType::FUNC_INDEX_PUT] = new CPUIndexPutFunction();
    map_function_[TensorFuncType::FUNC_SPLIT] = new CPUsplitFunction();
    map_function_[TensorFuncType::FUNC_SUM] = new CPUsumFunction();
    map_function_[TensorFuncType::FUNC_TOPK] = new CPUtopkFunction();
    map_function_[TensorFuncType::FUNC_EXPPAND] = new CPUexpandFunction();
    map_function_[TensorFuncType::FUNC_ARGSORT] = new CPUargsortFunction();
    map_function_[TensorFuncType::FUNC_BINCOUNT] = new CPUbincountFunction();
    map_function_[TensorFuncType::FUNC_REPEAT] = new CPUrepeatFunction();
    map_function_[TensorFuncType::FUNC_LIKE] = new CPUlikeFunction();
    map_function_[TensorFuncType::FUNC_SCATTERREDUCE] = new CPUScatterReduceFunction();
    map_function_[TensorFuncType::FUNC_APPLY_VISIOROPE] = new CPUApplyVisionRoPEFunction();
    // models use only
    map_function_[TensorFuncType::FUNC_FUYU_GATHER_EMBD] = new CPUFuyuGatherEmbdFunc();
    map_function_[TensorFuncType::FUNC_PHI3V_HD_MERGE] = new CPUPhi3VhdmergeFunction();
};

int CPUBackend::cpu_threads = 4;

} // namespace mllm
