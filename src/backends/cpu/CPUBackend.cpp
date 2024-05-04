#include "CPUBackend.hpp"
#include <iostream>
#include <math.h>
#include "CPUView.hpp"
#include "CPUAdd.hpp"
#include "CPUCausalMask.hpp"
#include "CPUSlidingWindowMask.hpp"
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
#include "CPUTensorFunction.hpp"

namespace mllm {
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
        printf("Don't support type \n");
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
    addCreator(EMBEDDING, (CPUBackend::Creator *)(new CPUEmbeddingCreator()));
    addCreator(MUL, (CPUBackend::Creator *)(new CPUMulCreator()));
    addCreator(VIEW, (CPUBackend::Creator *)(new CPUViewCreator()));
    addCreator(KVCACHE, (CPUBackend::Creator *)(new CPUKVCacheCreator()));
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
}

TensorFunction *CPUBackend::funcCreate(const TensorFuncType type) {
    auto iter = map_function_.find(type);
    if (iter == map_function_.end()) {
        printf("Don't support type \n");
        return nullptr;
    }
    return iter->second;
}

void CPUBackend::registerFuncs() {
    map_function_[TensorFuncType::FUNC_ADD] = new CPUaddFunction();
    map_function_[TensorFuncType::FUNC_SUB] = new CPUsubFunction();
    map_function_[TensorFuncType::FUNC_MUL] = new CPUmulFunction();
    map_function_[TensorFuncType::FUNC_DIV] = new CPUdivFunction();
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
    map_function_[TensorFuncType::FUNC_RANGE] = new CPURangeFunction();
    map_function_[TensorFuncType::FUNC_WHERE] = new CPUwhereFunction();
};

int CPUBackend::cpu_threads = 4;

} // namespace mllm
