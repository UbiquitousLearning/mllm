#include "CPUBackend.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
// #include "memory/SystemMemoryManager.hpp"
// #include <memory/MemoryPoolManager.hpp>
#include <string>
#include "Layer.hpp"

#include "op/CPUHeadLinear.hpp"
#include "op/CPULinearInt8.hpp"
#include "op/CPUMultimodalRoPEPipeline.hpp"
#include "op/CPUNTKRoPE.hpp"
// #include "op/CPUPoEmbedding.hpp"
#include "op/CPUSplitInput.hpp"
#include "op/CPUView.hpp"
#include "op/CPUAdd.hpp"
#include "op/CPUCausalMask.hpp"
#include "op/CPUCausalTreeMask.hpp"
#include "op/CPUSlidingWindowMask.hpp"
#include "op/CPUMatmul.hpp"
#include "op/CPURMSNorm.hpp"
#include "op/CPURoPE.hpp"
#include "op/CPURoPETree.hpp"
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
// #include "op/CPUCat.hpp"
#include "op/CPUSubDim.hpp"
#include "op/CPUQuickGELU.hpp"
#include "op/CPUDivision.hpp"
#include "op/CPUNorm.hpp"
#include "op/CPUShape.hpp"
#include "op/CPUTranspose.hpp"
#include "op/CPUMean.hpp"
#include "op/CPURange.hpp"
#include "op/CPUVisionRoPECos.hpp"
#include "op/CPUVisionRoPESin.hpp"
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
#include "op/CPUKVCacheSage.hpp"

#include "op/CPUBinaryFunc.hpp"
#include "op/CPUCatFunc.hpp"
#include "op/CPUClipFunc.hpp"
#include "op/CPUExpandFunc.hpp"
#include "op/CPUFlattenFunc.hpp"
#include "op/CPUMatmulFunc.hpp"
#include "op/CPUMeanFunc.hpp"
#include "op/CPUNormFunc.hpp"
#include "op/CPURangeFunc.hpp"
#include "op/CPUSplitFunc.hpp"
#include "op/CPUSumFunc.hpp"
#include "op/CPUTopkFunc.hpp"
#include "op/CPUTransposeFunc.hpp"
#include "op/CPUViewFunc.hpp"
#include "op/CPUWhereFunc.hpp"
#include "op/CPUIndexPutFunc.hpp"
#include "op/CPUArgSortFunc.hpp"
#include "op/CPUBinCountFunc.hpp"
#include "op/CPURepeatFunc.hpp"
#include "op/CPULikeFunc.hpp"
#include "op/CPUScatterAddFunc.hpp"
#include "op/CPUVisionRoPEFunc.hpp"
#include "op/CPUFlashAttention2Func.hpp"
#include "op/CPUSageAttentionFunc.hpp"
#include "op/CPUScatter.hpp"
#include "op/CPUTilde.hpp"
#include "op/CPUMaskedFill.hpp"
#include "op/CPUSigmoid.hpp"

#include "op/CPUFuyuGatherEmbdFunc.hpp"
#include "op/CPUPhi3VhdmergeFunc.hpp"

namespace mllm {
class CPUBackendCreator : public BackendCreator {
    Backend *create(BackendConfig config) {
        shared_ptr<MemoryManager> mm = nullptr;
        // mm = std::make_shared<SystemMemoryManager>();
        mm = std::make_shared<MemoryPoolManager>(); // todomm
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
    // registerFuncs();
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
    addCreator(CAUSALTREEMASK, (CPUBackend::Creator *)(new CPUCausalTreeMaskCreator()));
    addCreator(SLIDINGWINDOWMASK, (CPUBackend::Creator *)(new CPUSlidingWindowMaskCreator()));
    addCreator(MATMUL, (CPUBackend::Creator *)(new CPUMatmulCreator()));
    addCreator(RMSNORM, (CPUBackend::Creator *)(new CPURMSNormCreator()));
    addCreator(ROPE, (CPUBackend::Creator *)(new CPURoPECreator()));
    addCreator(ROPETREE, (CPUBackend::Creator *)(new CPURoPETreeCreator()));
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
    addCreator(VISIONROPESIN, (CPUBackend::Creator *)(new CPUVisionRoPESinCreator()));
    addCreator(VISIONROPECOS, (CPUBackend::Creator *)(new CPUVisionRoPECosCreator()));
    addCreator(MULTIMODALROPEPIP, (CPUBackend::Creator *)(new CPUMultimodalRoPEPipelineCreator()));
    addCreator(MULTIMODALROPE, (CPUBackend::Creator *)(new CPUMultimodalRoPECreator()));
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
    addCreator(KVCACHESAGE, (CPUBackend::Creator *)(new CPUKVCacheSageCreator()));
    addCreator(SIGMOID, (CPUBackend::Creator *)(new CPUSigmoidCreator()));

    // funsction
    addCreator(F_ADD, (CPUBackend::Creator *)(new CPUaddFunctionCreator()));
    addCreator(F_SUB, (CPUBackend::Creator *)(new CPUsubFunctionCreator()));
    addCreator(F_MUL, (CPUBackend::Creator *)(new CPUmulFunctionCreator()));
    addCreator(F_DIV, (CPUBackend::Creator *)(new CPUdivFunctionCreator()));
    addCreator(F_DIVINT, (CPUBackend::Creator *)(new CPUdivintFunctionCreator()));
    addCreator(F_TTADD, (CPUBackend::Creator *)(new CPUaddTwoFunctionCreator()));
    addCreator(F_TTSUB, (CPUBackend::Creator *)(new CPUsubTwoFunctionCreator()));
    addCreator(F_TTMUL, (CPUBackend::Creator *)(new CPUmulTwoFunctionCreator()));
    addCreator(F_TTDIV, (CPUBackend::Creator *)(new CPUdivTwoFunctionCreator()));
    addCreator(F_MM, (CPUBackend::Creator *)(new CPUmmFunctionCreator()));
    addCreator(F_NORM, (CPUBackend::Creator *)(new CPUnormFunctionCreator()));
    addCreator(F_MEAN, (CPUBackend::Creator *)(new CPUmeanFunctionCreator()));
    addCreator(F_CAT, (CPUBackend::Creator *)(new CPUcatFunctionCreator()));
    addCreator(F_VIEW, (CPUBackend::Creator *)(new CPUviewFunctionCreator()));
    addCreator(F_TRANPOSE, (CPUBackend::Creator *)(new CPUtransposeFunctionCreator()));
    addCreator(F_FLATTEN, (CPUBackend::Creator *)(new CPUflattenFunctionCreator()));
    addCreator(F_CLIP, (CPUBackend::Creator *)(new CPUclipFunctionCreator()));
    addCreator(F_CLIPAXIS, (CPUBackend::Creator *)(new CPUclipaxisFunctionCreator()));
    addCreator(F_CLIPTENSOR, (CPUBackend::Creator *)(new CPUcliptensorFunctionCreator()));
    addCreator(F_RANGE, (CPUBackend::Creator *)(new CPURangeFunctionCreator()));
    addCreator(F_WHERE, (CPUBackend::Creator *)(new CPUwhereFunctionCreator()));
    addCreator(F_INDEX_PUT, (CPUBackend::Creator *)(new CPUIndexPutFunctionCreator()));
    addCreator(F_SPLIT, (CPUBackend::Creator *)(new CPUsplitFunctionCreator()));
    addCreator(F_SUM, (CPUBackend::Creator *)(new CPUsumFunctionCreator()));
    addCreator(F_TOPK, (CPUBackend::Creator *)(new CPUtopkFunctionCreator()));
    addCreator(F_EXPPAND, (CPUBackend::Creator *)(new CPUexpandFunctionCreator()));
    addCreator(F_ARGSORT, (CPUBackend::Creator *)(new CPUargsortFunctionCreator()));
    addCreator(F_BINCOUNT, (CPUBackend::Creator *)(new CPUbincountFunctionCreator()));
    addCreator(F_REPEAT, (CPUBackend::Creator *)(new CPUrepeatFunctionCreator()));
    addCreator(F_LIKE, (CPUBackend::Creator *)(new CPUlikeFunctionCreator()));
    addCreator(F_SCATTERRADD, (CPUBackend::Creator *)(new CPUScatterAddFunctionCreator()));
    addCreator(F_APPLY_VISIOROPE, (CPUBackend::Creator *)(new CPUVisionRoPEFuncFunctionCreator()));
    addCreator(F_FA2, (CPUBackend::Creator *)(new CPUFlashAttention2FuncCreator()));
    addCreator(F_SAGEATTN, (CPUBackend::Creator *)(new CPUSageAttentionFuncCreator()));
    addCreator(SCATTER, (CPUBackend::Creator *)(new CPUScatterCreator()));
    addCreator(TILDE, (CPUBackend::Creator *)(new CPUTildeCreator()));
    addCreator(MASKEDFILL, (CPUBackend::Creator *)(new CPUMaskedFillCreator()));
    // models use only
    addCreator(F_FUYU_GATHER_EMBD, (CPUBackend::Creator *)(new CPUFuyuGatherEmbdFuncCreator()));
    addCreator(F_PHI3V_HD_MERGE, (CPUBackend::Creator *)(new CPUPhi3VhdmergeFunctionCreator()));
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
    ;
};

int CPUBackend::cpu_threads = 4;

void CPUBackend::convert_fp_data(Tensor *src, Tensor *dest) {
    // 根据源和目标的类型，执行相应的CPU循环转换
    if (src->dtype() == MLLM_TYPE_F32 && dest->dtype() == MLLM_TYPE_F16) {
        float *src_ptr = src->hostPtr<float>();
        mllm_fp16_t *dst_ptr = dest->hostPtr<mllm_fp16_t>();
        for (int i = 0; i < src->count(); i++) {
            dst_ptr[i] = MLLM_FP32_TO_FP16(src_ptr[i]);
        }
    } else if (src->dtype() == MLLM_TYPE_F16 && dest->dtype() == MLLM_TYPE_F32) {
        mllm_fp16_t *src_ptr = src->hostPtr<mllm_fp16_t>();
        float *dst_ptr = dest->hostPtr<float>();
        for (int i = 0; i < src->count(); i++) {
            dst_ptr[i] = MLLM_FP16_TO_FP32(src_ptr[i]);
        }
    } else {
        throw std::runtime_error("Unsupported conversion types for CPU backend.");
    }
}
/************************************************************************************************/
/* Refactored Helper Functions                                */
/************************************************************************************************/
/**
 * @brief Creates the initial output tensor objects (shells), either from an aggregated input or from a list of names.
 * @param out_tensors The vector of output tensors to be populated.
 * @param input_tensors The vector of input tensors, checked for aggregation.
 * @param out_names The names for the output tensors if not from an aggregated input.
 * @param module The current module.
 * @param backend The current backend.
 */
void CPUBackend::_create_output_tensors(
    std::vector<std::shared_ptr<Tensor>> &out_tensors,
    const std::vector<std::shared_ptr<Tensor>> &input_tensors,
    const std::vector<std::string> &out_names,
    Module *module,
    map<std::string, std::shared_ptr<Tensor>> &activation_tensors,
    Backend *backend) {
    if (input_tensors.size() == 1 && !input_tensors[0]->aggregatedTensors().empty()) {
        const auto &aggregated_tensors = input_tensors[0]->aggregatedTensors();
        out_tensors.insert(out_tensors.end(), aggregated_tensors.begin(), aggregated_tensors.end());
    } else {
        for (const auto &out_name : out_names) {
            auto out_tensor = std::make_shared<Tensor>(backend);
            out_tensor->setName(out_name);
            out_tensor->setModule(module);
            auto it = activation_tensors.find(out_name);
            if (it != activation_tensors.end() && out_name.find("-transpose") == std::string::npos && out_tensor->ctype() != it->second->ctype()) {
                out_tensor->chls() = it->second->chls();
                out_tensor->setCtype(it->second->ctype());
            }
            out_tensors.push_back(out_tensor);
        }
    }
}

std::vector<Tensor> CPUBackend::runOp(Op *op, std::vector<Tensor> inputs, std::vector<std::string> out_names, bool in_place) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    static map<string, shared_ptr<Tensor>> empty_activation_tensors;
    map<string, shared_ptr<Tensor>> &activation_tensors = module ? module->activation_tensors : empty_activation_tensors;
    if (module && module->doTrace) { // trace
        if (module->tracedFlag) {
            vector<Tensor> results = {};
            for (auto &name : out_names) results.push_back(*activation_tensors[name]);
            return results;
        }
        for (auto &input : inputs) {
            if (input.shouldInGraphs() && activation_tensors.find(input.name()) == activation_tensors.end()) {
                activation_tensors[input.name()] = std::make_shared<Tensor>(op->backend());
                activation_tensors[input.name()]->setName(input.name());
                activation_tensors[input.name()]->setModule(module);
            }
        }
        for (const auto &out_name : out_names) {
            if (activation_tensors.find(out_name) == activation_tensors.end()) {
                activation_tensors[out_name] = std::make_shared<Tensor>(op->backend());
                activation_tensors[out_name]->setName(out_name);
                activation_tensors[out_name]->setModule(module);
            }
        }
        vector<shared_ptr<Tensor>> inPtrs;
        for (auto &input : inputs) {
            inPtrs.push_back(input.shouldInGraphs() ? activation_tensors[input.name()] :
                                                      std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
        }
        vector<shared_ptr<Tensor>> outPtrs = {};
        for (auto &name : out_names) outPtrs.push_back(activation_tensors[name]);
        op->setUp(inPtrs, outPtrs);
        vector<Tensor> results = {};
        for (auto &name : out_names) results.push_back(*activation_tensors[name]);
        return results;
    }

#ifdef DEBUGOPTIME
    uint64_t time_start = mllm_time_us();
#endif
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
    }
    vector<shared_ptr<Tensor>> out_tensors;
    // Part 1: Create tensor shells
    if (!in_place) {
        _create_output_tensors(out_tensors, input_tensors, out_names, module, activation_tensors, op->backend());
    } else {
        // If in-place, we already have out_tensors filled with input tensors.
        for (size_t i = 0; i < input_tensors.size() && i < out_names.size(); ++i) {
            input_tensors[i]->setName(out_names[i]);
            out_tensors.push_back(input_tensors[i]);
        }
    }
    // Part 2: Reshape the tensors
    op->reshape(input_tensors, out_tensors);
    // Part 3: Allocate memory
    if (!in_place) {
        for (auto &out_tensor : out_tensors) {
            auto act_it = activation_tensors.find(out_tensor->name());
            auto template_it = act_it != activation_tensors.end() ? act_it->second : nullptr;
            out_tensor->allocFromTemplate(template_it);
        }
    }
    // Part 4: Execute the operation
    op->execute(input_tensors, out_tensors);

#ifdef DEBUGOPTIME
    uint64_t time_end = mllm_time_us();
    double inference_time_ = (time_end - time_start) / 1000.0F; // ms
    static int op_count = 0;
    if (op_inference_time_.empty()) {
        op_count = 0;
    }
    string name = std::to_string(op_count++) + "--" + (op->name().empty() ? (out_names.empty() ? "out-" + input_tensors[0]->name() : out_names[0]) : op->name());
    if (op->type() == LINEAR)
        op_inference_time_[name] = inference_time_;
#endif

    vector<Tensor> results;
    for (const auto &out_tensor : out_tensors) {
        results.push_back(*out_tensor);
#ifdef DEBUGSAVETENSOR
        if (out_tensor->dtype() == MLLM_TYPE_F32)
            out_tensor->saveData<float>();
        if (out_tensor->dtype() == MLLM_TYPE_F16)
            out_tensor->saveData<mllm_fp16_t>();
#endif
    }
    return results;
}

std::vector<Tensor> CPUBackend::runLayer(Layer *layer, std::vector<Tensor> inputs, int N) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    vector<string> out_names;
    int count = (N > 1) ? N : 1;
    for (int i = 0; i < count; ++i) {
        std::string tensor_name = (N > 1) ? "out-" + layer->op_->name() + "-" + std::to_string(i) : "out-" + layer->op_->name();
        out_names.push_back(tensor_name);
    }
    return runOp(layer->op_, inputs, out_names, false);
}

std::vector<Tensor> CPUBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    if (mllm::Module::llm_model_ptr && (mllm::Module::llm_model_ptr->doLoad || Module::llm_model_ptr->doChangeBn)) {
        auto outputs = module->Forward(inputs, args);
        return outputs;
    }
    uint64_t time_start, time_end;
    bool ouilter_flag = (inputs[0].ttype() == TensorType::INPUT_TENSOR);
    if (ouilter_flag) {
        for (int i = 0; i < inputs.size(); i++) {
            auto &input = inputs[i];
            input.setModule(module);
            input.setTtype(TensorType::NORMAL_TENSOR);
        }
        mllm::Module::llm_model_ptr = module;
        if (module->prefilling_token_size_ == 0) { // first time init
            module->prefilling_token_size_ = inputs[0].sequence() * inputs[0].batch();
        } else if (module->decoding_token_size_ == 0) {
            module->decoding_token_size_ = inputs[0].sequence() * inputs[0].batch();
        }
        time_start = mllm_time_us();
#ifdef DEBUGOPTIME
        op_inference_time_.clear();
#endif
    }

    auto output = module->Forward(inputs, args);

    if (ouilter_flag) {
        time_end = mllm_time_us();
        double inference_time_ = (time_end - time_start) / 1000.0F; // ms
        module->inference_times_.push_back(inference_time_);
#ifdef DEBUGOPTIME
        _print_op_inference_time(true);
        std::cout << "Token inference e2e time: " << inference_time_ << "ms" << std::endl;
#endif
    }
    return output;
}

void CPUBackend::_print_op_inference_time(bool sort) {
    size_t max_len = 0;
    for (const auto &pair : op_inference_time_) {
        max_len = std::max(pair.first.size(), max_len);
    }
    std::vector<std::pair<std::string, double>> sorted_pairs;
    if (sort) {
        sorted_pairs.assign(op_inference_time_.begin(), op_inference_time_.end());
        std::sort(sorted_pairs.begin(), sorted_pairs.end(),
                  [](const auto &a, const auto &b) {
                      return a.second > b.second;
                  });
    }
    double token_inference_time = 0.0;
    if (sort) {
        for (const auto &pair : sorted_pairs) {
            std::cout << std::left << std::setw(max_len) << pair.first
                      << " | time: " << pair.second << "ms" << std::endl;
            token_inference_time += pair.second;
        }
    } else {
        for (const auto &pair : op_inference_time_) {
            std::cout << std::left << std::setw(max_len) << pair.first
                      << " | time: " << pair.second << "ms" << std::endl;
            token_inference_time += pair.second;
        }
    }
    std::cout << "Op times sum: " << token_inference_time << "ms" << std::endl;
}

} // namespace mllm
