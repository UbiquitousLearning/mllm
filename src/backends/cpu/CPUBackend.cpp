#include "CPUBackend.hpp"
#include <iostream>
#include <math.h>
#include <memory>
#include "Backend.hpp"
#include "OpDefined.hpp"
#include "Types.hpp"
#include "memory/SystemMemoryManager.hpp"
#include <memory/MemoryPoolManager.hpp>
#include "Layer.hpp"

#include "op/CPUHeadLinear.hpp"
#include "op/CPULinearInt8.hpp"
#include "op/CPUMultimodalRoPEPipeline.hpp"
#include "op/CPUNTKRoPE.hpp"
#include "op/CPUPoEmbedding.hpp"
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
#include "function/CPUVisionRoPEFunc.hpp"
#include "function/CPUFlashAttention2Func.hpp"

#include "function/CPUFuyuGatherEmbdFunc.hpp"
#include "function/CPUPhi3VhdmergeFunc.hpp"

namespace mllm {
class CPUBackendCreator : public BackendCreator {
    Backend *create(BackendConfig config) {
        shared_ptr<MemoryManager> mm = nullptr;
        mm = std::make_shared<MemoryPoolManager>(); // todomm
        // switch (config.memory) {
        // case BackendConfig::Memory_High:
        //     mm = std::make_shared<SystemMemoryManager>();
        //     // mm = std::make_shared<MemoryPoolManager>(); // todomm
        //     break;
        // default:
        //     mm = std::make_shared<SystemMemoryManager>();
        //     // mm = std::make_shared<MemoryPoolManager>(); // todomm
        //     break;
        // }
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
    addCreator(MULTIMODALROPEPIP, (CPUBackend::Creator *)(new CPUMultimodalRoPEPipelineCreator()));
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
    map_function_[TensorFuncType::FUNC_APPLY_VISIOROPE] = new CPUVisionRoPEFuncFunction();
    map_function_[TensorFuncType::FUNC_FA2] = new CPUFlashAttention2Func();
    // models use only
    map_function_[TensorFuncType::FUNC_FUYU_GATHER_EMBD] = new CPUFuyuGatherEmbdFunc();
    map_function_[TensorFuncType::FUNC_PHI3V_HD_MERGE] = new CPUPhi3VhdmergeFunction();
};

int CPUBackend::cpu_threads = 4;

std::vector<Tensor> CPUBackend::runFunc(
    std::vector<std::string> out_names,
    TensorFuncType type,
    std::vector<float> float_args,
    std::vector<std::shared_ptr<Tensor>> input_tensors,
    bool in_place) {
    Module *module = input_tensors.empty() ? Module::llm_model_ptr : input_tensors[0]->module();
    auto &activation_tensors = module->activation_tensors;
    assert(module != nullptr);
    Backend *backend = input_tensors.empty() ? Backend::global_backends[MLLM_CPU] : input_tensors[0]->backend();
    TensorFunction *func = backend->funcCreate(type);

    if (module->doLoad) {
        auto &activation_tensors_num = module->activation_tensors_num;
        for (const auto &out_name : out_names) {
            if (activation_tensors.find(out_name) == activation_tensors.end()) {
                activation_tensors[out_name] = std::make_shared<Tensor>(backend);
                activation_tensors[out_name]->setName(out_name);
                activation_tensors[out_name]->setModule(module);
                activation_tensors_num[out_name] = 0;
            }
        }
        std::vector<std::shared_ptr<Tensor>> inPtrs;
        for (auto &t : input_tensors) inPtrs.push_back(activation_tensors[t->name()]);
        std::vector<std::shared_ptr<Tensor>> outPtrs;
        for (auto &name : out_names) outPtrs.push_back(activation_tensors[name]);
        func->setUp(outPtrs, inPtrs, float_args);
        std::vector<Tensor> results;
        for (auto &name : out_names) results.push_back(*activation_tensors[name]);
        return results;
    }

#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    std::vector<std::shared_ptr<Tensor>> out_tensors;
    if (in_place) {
        for (size_t i = 0; i < input_tensors.size() && i < out_names.size(); ++i) {
            input_tensors[i]->setName(out_names[i]);
            out_tensors.push_back(input_tensors[i]);
        }
    } else {
        if (input_tensors.size() == 1 && !input_tensors[0]->aggregatedTensors().empty()) {
            auto aggregatedTensorsSize = input_tensors[0]->aggregatedTensors().size();
            for (int i = 0; i < aggregatedTensorsSize; i++) {
                out_tensors.push_back(input_tensors[0]->aggregatedTensors()[i]);
            }
        } else {
            for (auto out_name : out_names) {
                auto out_tensor = std::make_shared<Tensor>(backend);
                out_tensor->setName(out_name);
                out_tensor->setModule(module);
                auto it = activation_tensors.find(out_name);
                if (it != activation_tensors.end() && out_tensor->name().find("-transpose") == std::string::npos
                    && out_tensor->ctype() != it->second->ctype()) {
                    out_tensor->chls() = it->second->chls();
                    out_tensor->setCtype(it->second->ctype());
                }
                out_tensors.push_back(out_tensor);
            }
        }
    }
    func->reshape(out_tensors, input_tensors, float_args);
    for (auto &out_tensor : out_tensors) {
        if (activation_tensors.find(out_tensor->name()) != activation_tensors.end()
            && !activation_tensors[out_tensor->name()]->aggregatedTensors().empty()) {
            // 存在aggregatedTensors

            bool remove_f = true;
            if (activation_tensors[out_tensor->name()]->aggregatedDim() > 3) {
                remove_f = false; // 不能移除
            } else {
                for (auto &agtensor : activation_tensors[out_tensor->name()]->aggregatedTensors()) {
                    bool check_m = (agtensor->masterTensor() != nullptr
                                    && agtensor->masterTensor()->name().find("Cache") != std::string::npos);
                    if (agtensor->ctype() != activation_tensors[out_tensor->name()]->aggregatedTensors()[0]->ctype()) {
                        remove_f = false;
                    } else if (check_m) {
                        remove_f = false;
                    }
                }
            }
            // if (remove_f) {
            //     inputs[0]->removeAggregatedTensors();
            //     inputs[0]->allowAggregated() = false;
            // }
            if (!remove_f) {
                vector<shared_ptr<Tensor>> shared_outputs = {};
                auto split_dim = activation_tensors[out_tensor->name()]->aggregatedDim();
                for (int id = 0; id < activation_tensors[out_tensor->name()]->aggregatedTensors().size(); id++) {
                    auto shared_ot = std::make_shared<Tensor>(backend);
                    shared_ot->setName(out_tensor->name() + ".split-" + std::to_string(id));
                    shared_ot->setModule(module);
                    auto ot = activation_tensors[out_tensor->name()]->aggregatedTensors()[id];
                    shared_ot->setCtype(ot->ctype());
                    switch (split_dim) {
                    case Chl::HEAD: {
                        shared_ot->reshape(out_tensor->batch(), ot->head(), out_tensor->sequence(), out_tensor->dimension());
                        break;
                    }
                    case Chl::SEQUENCE: {
                        shared_ot->reshape(out_tensor->batch(), out_tensor->head(), ot->sequence(), out_tensor->dimension());
                        break;
                    }
                    case Chl::DIMENSION: {
                        shared_ot->reshape(out_tensor->batch(), out_tensor->head(), out_tensor->sequence(), ot->dimension());
                        break;
                    }
                    case Chl::D_HD:
                    case Chl::HD: {
                        shared_ot->reshape(out_tensor->batch(), ot->head(), out_tensor->sequence(), ot->dimension());
                        break;
                    }
                    default: {
                        break;
                    }
                    }
                    if (activation_tensors[shared_ot->name()]->masterTensor() != nullptr
                        && activation_tensors[shared_ot->name()]->masterTensor()->name().find("Cache") != std::string::npos) {
                        auto cache_seq_len_ = activation_tensors[shared_ot->name()]->shapeOffset()[2];
                        if (shared_ot->name().find("cache") == std::string::npos) { // KVcahe的输出不设置，只有输入设置
                            cache_seq_len_ = activation_tensors[shared_ot->name()]->masterTensor()->cache_seq_len_;
                            auto cpuBackend = dynamic_cast<CPUBackend *>(backend);
                            if (cpuBackend->isUsingDraft()) {
                                unsigned int last_draft_length = cpuBackend->getLastDraftLength();
                                const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
                                cache_seq_len_ = cache_seq_len_ - (last_draft_length) + last_verified_position_ids.size();
                            }
                        }
                        shared_ot->setDtype(activation_tensors[shared_ot->name()]->masterTensor()->dtype());
                        // masterTensor() 是Cache所以shape没有问题
                        shared_ot->shallowCopyFrom(activation_tensors[shared_ot->name()]->masterTensor(), false, {0, 0, cache_seq_len_, 0});
                    } else {
                        shared_ot->alloc();
                    }
                    shared_outputs.push_back(shared_ot);
                }
                out_tensor->addTensors(shared_outputs, split_dim);
            } else {
                out_tensor->allowAggregated() = false;
                out_tensor->alloc();
            }
        } else if (activation_tensors.find(out_tensor->name()) != activation_tensors.end()
                   && activation_tensors[out_tensor->name()]->masterTensor() != nullptr
                   && activation_tensors[out_tensor->name()]->masterTensor()->name().find("Cache") != std::string::npos) {
            // output_tensor的master是KVCache
            auto cache_seq_len_ = activation_tensors[out_tensor->name()]->shapeOffset()[2];
            if (out_tensor->name().find("cache") == std::string::npos) { // KVcahe的输出不设置，只有输入设置
                cache_seq_len_ = activation_tensors[out_tensor->name()]->masterTensor()->cache_seq_len_;
                auto cpuBackend = dynamic_cast<CPUBackend *>(backend);
                if (cpuBackend->isUsingDraft()) {
                    unsigned int last_draft_length = cpuBackend->getLastDraftLength();
                    const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
                    cache_seq_len_ = cache_seq_len_ - (last_draft_length) + last_verified_position_ids.size();
                }
            }
            out_tensor->setDtype(activation_tensors[out_tensor->name()]->masterTensor()->dtype());
            out_tensor->shallowCopyFrom(activation_tensors[out_tensor->name()]->masterTensor(), false, {0, 0, cache_seq_len_, 0});
        } else {
            out_tensor->alloc();
        }
    }
    func->execute(out_tensors, input_tensors, float_args);
#ifdef DEBUGOPTIME
    auto end_t = mllm_time_us();
    std::cout << out_names[0] << " |  time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
#endif
    vector<Tensor> results;
    for (const auto &out_tensor : out_tensors) { results.push_back(*out_tensor); }
    return results;
}

std::vector<Tensor> CPUBackend::runLayer(Layer *layer, std::vector<Tensor> inputs, int N) {
    Module *module = inputs.empty() ? Module::llm_model_ptr : inputs[0].module();
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    bool do_init = false;
    if (module->doLoad || !layer->inited_loaded) {
        // set backend to current module device and try to create op
        // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
        layer->backend_ = Backend::global_backends[MLLM_CPU];
        do_init = !layer->inited_loaded;
        if (layer->op_ == nullptr) {
            layer->op_ = layer->backend_->opCreate(layer->param_, layer->name_);
        }
        if (module->doLoad) {
            layer->op_->load(*module->loader);
            layer->inited_loaded = true;
        } else if (layer->loaded_param) {
            layer->inited_loaded = layer->loaded_param;
        } else {
            if (!layer->inited_loaded) {
                auto empty_loader = new ParamLoader("");
                layer->op_->load(*empty_loader);
                layer->inited_loaded = true;
            }
        }
        vector<string> out_names = {};
        int count = (N > 1) ? N : 1;
        for (int i = 0; i < count; ++i) {
            std::string out_name = (N > 1) ? "out-" + layer->op_->name() + "-" + std::to_string(i) : "out-" + layer->op_->name();
            out_names.push_back(out_name);
            if (activation_tensors.find(out_name) == activation_tensors.end()) {
                activation_tensors[out_name] = std::make_shared<Tensor>(layer->backend_);
                activation_tensors[out_name]->setName(out_name);
                activation_tensors[out_name]->setModule(module);
            }
        }
        if (module->doLoad) {
            // input_tensors
            vector<shared_ptr<Tensor>> inPtrs;
            for (auto &input : inputs) {
                inPtrs.push_back(input.shouldInGraphs() ? activation_tensors[input.name()] : std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
            }
            // output_tensors
            vector<shared_ptr<Tensor>> outPtrs = {};
            for (auto &name : out_names) outPtrs.push_back(activation_tensors[name]);
            layer->op_->setUp(inPtrs, outPtrs);
            vector<Tensor> results = {};
            for (auto &name : out_names) results.push_back(*activation_tensors[name]);
            return results;
        }
    }
    // NEW START

#ifdef DEBUGOPTIME
    uint64_t time_start = mllm_time_us();
#endif
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
    }
    vector<shared_ptr<Tensor>> out_tensors;
    if (input_tensors.size() == 1 && !input_tensors[0]->aggregatedTensors().empty()) {
        auto aggregatedTensorsSize = input_tensors[0]->aggregatedTensors().size();
        for (int i = 0; i < aggregatedTensorsSize; i++) {
            out_tensors.push_back(input_tensors[0]->aggregatedTensors()[i]);
        }
    } else {
        int count = (N > 1) ? N : 1;
        for (int i = 0; i < count; ++i) {
            std::string tensor_name = (N > 1) ? "out-" + layer->op_->name() + "-" + std::to_string(i) : "out-" + layer->op_->name();
            auto out_tensor = std::make_shared<Tensor>(layer->backend_);
            out_tensor->setName(tensor_name);
            out_tensor->setModule(module);
            if (out_tensor->name().find("-transpose") == std::string::npos
                && out_tensor->ctype() != activation_tensors.at(out_tensor->name())->ctype()) {
                out_tensor->chls() = activation_tensors.at(out_tensor->name())->chls();
                out_tensor->setCtype(activation_tensors.at(out_tensor->name())->ctype());
            }
            out_tensors.push_back(out_tensor);
        }
    }
    // 直接使用 out_tensors 进行 reshape
    layer->op_->reshape(input_tensors, out_tensors);
    // 直接使用 out_tensors 进行 alloc
    for (auto &out_tensor : out_tensors) {
        if (activation_tensors.find(out_tensor->name()) != activation_tensors.end()
            && !activation_tensors[out_tensor->name()]->aggregatedTensors().empty()) {
            // 存在aggregatedTensors
            bool remove_f = true;
            if (activation_tensors[out_tensor->name()]->aggregatedDim() > 3) {
                remove_f = false; // 不能移除
            } else {
                for (auto &agtensor : activation_tensors[out_tensor->name()]->aggregatedTensors()) {
                    bool check_m = (agtensor->masterTensor() != nullptr
                                    && agtensor->masterTensor()->name().find("Cache") != std::string::npos);
                    if (agtensor->ctype() != activation_tensors[out_tensor->name()]->aggregatedTensors()[0]->ctype()) {
                        remove_f = false;
                    } else if (check_m) {
                        remove_f = false;
                    }
                }
            }
            if (!remove_f) {
                vector<shared_ptr<Tensor>> shared_outputs = {};
                auto split_dim = activation_tensors[out_tensor->name()]->aggregatedDim();
                for (int id = 0; id < activation_tensors[out_tensor->name()]->aggregatedTensors().size(); id++) {
                    auto shared_ot = std::make_shared<Tensor>(layer->backend_);
                    shared_ot->setName(out_tensor->name() + ".split-" + std::to_string(id));
                    shared_ot->setModule(module);
                    auto ot = activation_tensors[out_tensor->name()]->aggregatedTensors()[id];
                    shared_ot->setCtype(ot->ctype());
                    switch (split_dim) {
                    case Chl::HEAD: {
                        shared_ot->reshape(out_tensor->batch(), ot->head(), out_tensor->sequence(), out_tensor->dimension());
                        break;
                    }
                    case Chl::SEQUENCE: {
                        shared_ot->reshape(out_tensor->batch(), out_tensor->head(), ot->sequence(), out_tensor->dimension());
                        break;
                    }
                    case Chl::DIMENSION: {
                        shared_ot->reshape(out_tensor->batch(), out_tensor->head(), out_tensor->sequence(), ot->dimension());
                        break;
                    }
                    case Chl::D_HD:
                    case Chl::HD: {
                        shared_ot->reshape(out_tensor->batch(), ot->head(), out_tensor->sequence(), ot->dimension());
                        break;
                    }
                    default: {
                        break;
                    }
                    }
                    if (activation_tensors[shared_ot->name()]->masterTensor() != nullptr
                        && activation_tensors[shared_ot->name()]->masterTensor()->name().find("Cache") != std::string::npos) {
                        auto cache_seq_len_ = activation_tensors[shared_ot->name()]->shapeOffset()[2];
                        if (shared_ot->name().find("cache") == std::string::npos) { // KVcahe的输出不设置，只有输入设置
                            cache_seq_len_ = activation_tensors[shared_ot->name()]->masterTensor()->cache_seq_len_;
                            auto cpuBackend = dynamic_cast<CPUBackend *>(layer->backend_);
                            if (cpuBackend->isUsingDraft()) {
                                unsigned int last_draft_length = cpuBackend->getLastDraftLength();
                                const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
                                cache_seq_len_ = cache_seq_len_ - (last_draft_length) + last_verified_position_ids.size();
                            }
                        }
                        shared_ot->setDtype(activation_tensors[shared_ot->name()]->masterTensor()->dtype());
                        // masterTensor() 是Cache所以shape没有问题
                        shared_ot->shallowCopyFrom(activation_tensors[shared_ot->name()]->masterTensor(), false, {0, 0, cache_seq_len_, 0});
                    } else {
                        shared_ot->alloc();
                    }
                    shared_outputs.push_back(shared_ot);
                }
                out_tensor->addTensors(shared_outputs, split_dim);
            } else {
                out_tensor->allowAggregated() = false;
                out_tensor->alloc();
            }
        } else if (activation_tensors.find(out_tensor->name()) != activation_tensors.end()
                   && activation_tensors[out_tensor->name()]->masterTensor() != nullptr
                   && activation_tensors[out_tensor->name()]->masterTensor()->name().find("Cache") != std::string::npos) {
            // output_tensor的master是KVCache
            auto cache_seq_len_ = activation_tensors[out_tensor->name()]->shapeOffset()[2];
            if (out_tensor->name().find("cache") == std::string::npos) { // KVcahe的输出不设置，只有输入设置
                cache_seq_len_ = activation_tensors[out_tensor->name()]->masterTensor()->cache_seq_len_;
                auto cpuBackend = dynamic_cast<CPUBackend *>(layer->backend_);
                if (cpuBackend->isUsingDraft()) {
                    unsigned int last_draft_length = cpuBackend->getLastDraftLength();
                    const std::vector<unsigned int> &last_verified_position_ids = cpuBackend->getLastVerifiedPositionIds();
                    cache_seq_len_ = cache_seq_len_ - (last_draft_length) + last_verified_position_ids.size();
                }
            }
            out_tensor->setDtype(activation_tensors[out_tensor->name()]->masterTensor()->dtype());
            out_tensor->shallowCopyFrom(activation_tensors[out_tensors[0]->name()]->masterTensor(), false, {0, 0, cache_seq_len_, 0});
        } else {
            out_tensor->setDtype(MLLM_TYPE_F32);
            out_tensor->alloc();
        }
    }
    // 直接使用 out_tensors 进行 execute
    layer->op_->execute(input_tensors, out_tensors);

#ifdef DEBUGOPTIME
    uint64_t time_end = mllm_time_us();
    double inference_time_ = (time_end - time_start) / 1000.0F; // ms
    std::cout << op_->name() << " | time: " << inference_time_ << "ms" << std::endl;
#endif
    // 将 shared_ptr<Tensor> 转换为 Tensor 返回
    vector<Tensor> results;
    for (const auto &out_tensor : out_tensors) { results.push_back(*out_tensor); }
    return results;
}
std::vector<Tensor> CPUBackend::runForward(Module *module, std::vector<Tensor> inputs, std::vector<std::any> args) {
    if (mllm::Module::llm_model_ptr && mllm::Module::llm_model_ptr->doLoad) {
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
            module->prefilling_token_size_ = inputs[0].sequence();
        } else if (module->decoding_token_size_ == 0) {
            module->decoding_token_size_ = inputs[0].sequence();
        }
        time_start = mllm_time_us();
    }

    // Module setUp & execute
    auto output = module->Forward(inputs, args);

    if (ouilter_flag) {
        time_end = mllm_time_us();
        double inference_time_ = (time_end - time_start) / 1000.0F; // ms
        module->inference_times_.push_back(inference_time_);
        mllm::Module::llm_model_ptr->op_transposed_flag = true;
    }
    return output;
}
} // namespace mllm
