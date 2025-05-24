//
// Created by Rongjie Yi on 2024/2/4 0004.
//

#include "Layer.hpp"
namespace mllm {
map<string, string> Layer::layername_2_tensorname;
bool Layer::use_layername_2_tensorname = false;

std::shared_ptr<Tensor> createOutTensor(const std::string &name, Module *module, Backend *backend,
                                        const std::map<std::string, std::shared_ptr<Tensor>> &activation_tensors) {
    auto out_tensor = std::make_shared<Tensor>(backend);
    out_tensor->setName(name);
    out_tensor->setModule(module);
    if (out_tensor->name().find("-transpose") == std::string::npos && out_tensor->ctype() != activation_tensors.at(out_tensor->name())->ctype()) {
        out_tensor->chls() = activation_tensors.at(out_tensor->name())->chls();
        out_tensor->setCtype(activation_tensors.at(out_tensor->name())->ctype());
    }
    return out_tensor;
}
vector<Tensor> Layer::run(vector<Tensor> inputs, int N) {
    Module *module;
    if (!inputs.empty()) {
        module = inputs[0].module();
    } else {
        module = Module::llm_model_ptr;
    }
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    bool do_init = false;
    if (module->doLoad || !inited_loaded) {
        // set backend to current module device and try to create op
        // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
        backend_ = Backend::global_backends[Module::tmp_device];
        do_init = !inited_loaded;
        if (op_ == nullptr) {
            op_ = backend_->opCreate(param_, name_);
        }
        if (module->doLoad) {
            op_->load(*module->loader);
            inited_loaded = true;
        } else if (loaded_param) {
            inited_loaded = loaded_param;
        } else {
            if (!inited_loaded) {
                auto empty_loader = new ParamLoader("");
                op_->load(*empty_loader);
                inited_loaded = true;
            }
        }
        vector<string> layer_next_names = {};
        if (N > 1) {
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
            }
        } else {
            layer_next_names = {"out-" + op_->name()};
        }
        for (const auto &layer_next_name : layer_next_names) {
            if (activation_tensors.find(layer_next_name) == activation_tensors.end()) {
                activation_tensors[layer_next_name] = std::make_shared<Tensor>(backend_);
                activation_tensors[layer_next_name]->setName(layer_next_name);
                activation_tensors[layer_next_name]->setModule(module);
            }
        }
        if (module->doLoad) {
            // input_tensors
            vector<shared_ptr<Tensor>> input_tensors;
            for (auto &input : inputs) {
                if (input.shouldInGraphs()) {
                    auto input_name = input.name();
                    input_tensors.push_back(activation_tensors[input_name]);
                } else {
                    input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
                }
            }
            // output_tensors
            vector<shared_ptr<Tensor>> output_tensors = {};
            for (const auto &layer_next_name : layer_next_names) {
                output_tensors.push_back(activation_tensors[layer_next_name]);
            }
            op_->setUp(input_tensors, output_tensors);
            vector<Tensor> output_result = {};
            for (const auto &layer_next_name : layer_next_names) {
                output_result.push_back(*activation_tensors[layer_next_name]);
            }
            return output_result;
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
    int count = (N > 1) ? N : 1;
    for (int i = 0; i < count; ++i) {
        std::string tensor_name = (N > 1) ? "out-" + op_->name() + "-" + std::to_string(i) : "out-" + op_->name();
        auto out_tensor = createOutTensor(tensor_name, module, backend_, activation_tensors);
        out_tensors.push_back(out_tensor);
    }
    // 直接使用 out_tensors 进行 reshape
    op_->reshape(input_tensors, out_tensors);
    // 直接使用 out_tensors 进行 alloc
    if (activation_tensors.find(out_tensors[0]->name()) != activation_tensors.end()
        && out_tensors.size() == 1 && !activation_tensors[out_tensors[0]->name()]->aggregatedTensors().empty()) {
        // 存在aggregatedTensors
        vector<shared_ptr<Tensor>> shared_outputs = {};
        auto split_dim = activation_tensors[out_tensors[0]->name()]->aggregatedDim();
        for (int id = 0; id < activation_tensors[out_tensors[0]->name()]->aggregatedTensors().size(); id++) {
            auto shared_ot = std::make_shared<Tensor>(backend_);
            shared_ot->setName(out_tensors[0]->name() + ".split-" + std::to_string(id));
            shared_ot->setModule(module);
            auto ot = activation_tensors[out_tensors[0]->name()]->aggregatedTensors()[id];
            shared_ot->setCtype(ot->ctype());
            switch (split_dim) {
            case Chl::HEAD: {
                shared_ot->reshape(out_tensors[0]->batch(), ot->head(), out_tensors[0]->sequence(), out_tensors[0]->dimension());
                break;
            }
            case Chl::SEQUENCE: {
                shared_ot->reshape(out_tensors[0]->batch(), out_tensors[0]->head(), ot->sequence(), out_tensors[0]->dimension());
                break;
            }
            case Chl::DIMENSION: {
                shared_ot->reshape(out_tensors[0]->batch(), out_tensors[0]->head(), out_tensors[0]->sequence(), ot->dimension());
                break;
            }
            case Chl::D_HD:
            case Chl::HD: {
                shared_ot->reshape(out_tensors[0]->batch(), ot->head(), out_tensors[0]->sequence(), ot->dimension());
                break;
            }
            default: {
                break;
            }
            }
            if (activation_tensors[shared_ot->name()]->masterTensor() != nullptr && activation_tensors[shared_ot->name()]->masterTensor()->name().find("Cache") != std::string::npos) {
                auto cache_seq_len_ = activation_tensors[shared_ot->name()]->shapeOffset()[2];
                if (shared_ot->name().find("cache") == std::string::npos) { // KVcahe的输出不设置，只有输入设置
                    cache_seq_len_ = activation_tensors[shared_ot->name()]->masterTensor()->cache_seq_len_;
                }
                shared_ot->setDtype(activation_tensors[shared_ot->name()]->masterTensor()->dtype());
                // masterTensor() 是Cache所以shape没有问题
                shared_ot->shallowCopyFrom(activation_tensors[shared_ot->name()]->masterTensor(), false, {0, 0, cache_seq_len_, 0});
            } else {
                shared_ot->alloc();
            }
            shared_outputs.push_back(shared_ot);
        }
        out_tensors[0]->addTensors(shared_outputs, split_dim);
    } else if (activation_tensors.find(out_tensors[0]->name()) != activation_tensors.end()
               && out_tensors.size() == 1 && out_tensors[0]->masterTensor() == nullptr
               && activation_tensors[out_tensors[0]->name()]->masterTensor() != nullptr
               && activation_tensors[out_tensors[0]->name()]->masterTensor()->name().find("Cache") != std::string::npos) {
        // For KVCache
        auto cache_seq_len_ = activation_tensors[out_tensors[0]->name()]->shapeOffset()[2];
        if (out_tensors[0]->name().find("cache") == std::string::npos) { // KVcahe的输出不设置，只有输入设置
            cache_seq_len_ = activation_tensors[out_tensors[0]->name()]->masterTensor()->cache_seq_len_;
        }
        out_tensors[0]->setDtype(activation_tensors[out_tensors[0]->name()]->masterTensor()->dtype());
        out_tensors[0]->shallowCopyFrom(activation_tensors[out_tensors[0]->name()]->masterTensor(), false, {0, 0, cache_seq_len_, 0});
    } else {
        for (auto &output : out_tensors) {
            output->setDtype(MLLM_TYPE_F32);
            output->alloc();
        }
    }
    // 直接使用 out_tensors 进行 execute
    op_->execute(input_tensors, out_tensors);

#ifdef DEBUGOPTIME
    uint64_t time_end = mllm_time_us();
    double inference_time_ = (time_end - time_start) / 1000.0F; // ms
    std::cout << op_->name() << " | time: " << inference_time_ << "ms" << std::endl;
#endif
    // 将 shared_ptr<Tensor> 转换为 Tensor 返回
    vector<Tensor> output_result;
    // if (!input_tensors.empty())
    //     input_tensors[0]->saveData<float>();
    for (const auto &out_tensor : out_tensors) {
        // out_tensor->saveData<float>();
        output_result.push_back(*out_tensor);
    }
    return output_result;
}

// not Eager mod
/*
std::string name_num_to_X(const std::string &input_string) {
    std::regex pattern(R"(\.\d{1,3}\.)"); // Matches any number between 1 and 100 between two dots
    std::string replacement = ".X.";      // The string to replace the matched pattern with
    std::string output_string = std::regex_replace(input_string, pattern, replacement);
    return output_string;
}
std::string name_X_to_num(const std::string &input_string, int in_idx) {
    std::regex pattern(".X.");                                    // Matches any number between 1 and 100 between two dots
    std::string replacement = "." + std::to_string(in_idx) + "."; // The string to replace the matched pattern with
    std::string output_string = std::regex_replace(input_string, pattern, replacement);
    return output_string;
}
void init_reset_KVCache(string input_name, Module *module, int saved_list_idx, map<string, string> layername_2_tensorname, Backend *backend_) {
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    vector<string> renameX_names;
    renameX_names.push_back(input_name);
    const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2", "-cat", "-split-0-48"};
    vector<string> new_names;
    bool can_break = true;
    auto in_x_name = renameX_names[0];
    while (can_break) {
        can_break = false;
        for (const auto &suffix : suffixs) {
            if (in_x_name.rfind(suffix) == (in_x_name.size() - suffix.size())) {
                const auto r_name = in_x_name.substr(0, in_x_name.size() - suffix.size());
                if (std::find(renameX_names.begin(), renameX_names.end(), r_name) == renameX_names.end() && std::find(new_names.begin(), new_names.end(), r_name) == new_names.end()) {
                    new_names.push_back(r_name);
                    in_x_name = r_name;
                    can_break = true;
                }
                break;
            }
        }
    }
    renameX_names.insert(renameX_names.end(), new_names.begin(), new_names.end());
    for (const auto x_name : renameX_names) {
        auto name = name_X_to_num(x_name, saved_list_idx);
        layername_2_tensorname[name] = name;
        activation_tensors[name] = std::make_shared<Tensor>(backend_);
        activation_tensors[name]->initFrom(*activation_tensors[x_name]);
        activation_tensors[name]->setName(name);
        activation_tensors[name]->setModule(module);
    }
}

vector<Tensor> Layer::runSta(vector<Tensor> inputs, int N) {
    Module *module;
    if (!inputs.empty()) {
        module = inputs[0].module();
    } else {
        module = Module::llm_model_ptr;
    }
    map<string, shared_ptr<Tensor>> &activation_tensors = module->activation_tensors;
    auto &activation_tensors_num = module->activation_tensors_num;
    // Module::runlistIdx = saved_list_idx;
    bool do_init = false;

    if (module->doLoad || !inited_loaded) {
        // set backend to current module device and try to create op
        // use Module::tmp_device only when creating the op as the recersive module backend only handled in load and init stage
        backend_ = Backend::global_backends[Module::tmp_device];
        do_init = !inited_loaded;
        if (op_ == nullptr) {
#ifdef USE_QNN
            if ((param_["type"] == KVCACHE || param_["type"] == KVCACHENPU) && (Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end())) {
                if (kv_cache_map.find(name_) == kv_cache_map.end()) {
                    // for the prefill part, we need to create a new op
                    param_["type"] = KVCACHENPU;
                    op_ = backend_->opCreate(param_, name_);
                    kv_cache_map[name_] = op_;
                } else {
#ifdef DEBUGPRINT
                    std::cout << name_ << " is shared used" << std::endl;
#endif
                    // for the decoding part, we need to get created op from global container
                    op_ = kv_cache_map[name_];
                }
            } else {
                op_ = backend_->opCreate(param_, name_);
            }
#else
            op_ = backend_->opCreate(param_, name_);
#endif
        }
        if (module->doLoad) {
            op_->load(*module->loader);
            inited_loaded = true;
        } else if (loaded_param) {
            inited_loaded = loaded_param;
        } else {
            if (!inited_loaded) {
                // module->loader = new ParamLoader("");
                // op_->load(*module->loader);
                auto empty_loader = new ParamLoader("");
                op_->load(*empty_loader);
                inited_loaded = true;
            }
        }
        vector<string> layer_next_names = {};
        if (N > 1) {
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
            }
        } else {
            layer_next_names = {"out-" + op_->name()};
        }
        for (const auto &layer_next_name : layer_next_names) {
            string next_name;
            if (use_layername_2_tensorname) {
                if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                    if (param_["type"] == KVCACHE) {
                        layername_2_tensorname[layer_next_name] = layer_next_name;
                        init_reset_KVCache(inputs[0].name(), module, saved_list_idx, layername_2_tensorname, backend_);
                    } else {
                        layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                    }
                }
                next_name = layername_2_tensorname[layer_next_name];
            } else {
                next_name = layer_next_name;
            }
            if (activation_tensors.find(next_name) == activation_tensors.end()) {
                activation_tensors[next_name] = std::make_shared<Tensor>(backend_);
                activation_tensors[next_name]->setName(next_name);
                activation_tensors[next_name]->setModule(module);
                activation_tensors_num[next_name] = 0;
            }
        }
        if (module->doLoad) {
            // input_tensors
            vector<shared_ptr<Tensor>> input_tensors;
            for (auto &input : inputs) {
                if (input.shouldInGraphs()) {
                    auto input_name = input.name();
                    if (param_["type"] == KVCACHE && do_init && use_layername_2_tensorname) {
                        input_name = name_X_to_num(input_name, saved_list_idx);
                    }
                    input_tensors.push_back(activation_tensors[input_name]);
                } else {
                    input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
                }
            }
            // output_tensors
            vector<shared_ptr<Tensor>> output_tensors = {};
            for (const auto &layer_next_name : layer_next_names) {
                string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
                output_tensors.push_back(activation_tensors[next_name]);
            }
            op_->setUp(input_tensors, output_tensors);

            vector<Tensor> output_result = {};
            for (const auto &layer_next_name : layer_next_names) {
                string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
                output_result.push_back(*activation_tensors[next_name]);
            }
            return output_result;
        }
    }
    // input_tensors
    vector<shared_ptr<Tensor>> input_tensors;
    for (auto &input : inputs) {
        if (input.shouldInGraphs()) {
            auto input_name = input.name();
            if (param_["type"] == KVCACHE && do_init && use_layername_2_tensorname) {
                input_name = name_X_to_num(input_name, saved_list_idx);
            }
            input_tensors.push_back(activation_tensors[input_name]);
        } else {
            input_tensors.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
        }
    }
    // output_tensors
    vector<string> layer_next_names = {};
    if (N > 1) {
        for (int i = 0; i < N; ++i) {
            layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
        }
    } else {
        layer_next_names = {"out-" + op_->name()};
    }
    vector<shared_ptr<Tensor>> output_tensors = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
        output_tensors.push_back(activation_tensors[next_name]);
    }
#ifdef DEBUGOPTIME
    auto start_t = mllm_time_us();
#endif
    switch (Tensor::tensor_status) {
    case TENSOR_STATIC_INIT: {
        op_->reshape(input_tensors, output_tensors);
        op_->setUp(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_READY: {
        op_->execute(input_tensors, output_tensors);
        break;
    }
    case TENSOR_STATIC_TRACE: {
        if (backend_->type() == BackendType::MLLM_CPU) {
            Tracer::addOp(op_, input_tensors, output_tensors);
        }
        break;
    }
    default: {
        break;
    }
    }
    if (Backend::global_backends.size() == 1) {
        for (auto input_tensor : input_tensors) {
            if ((activation_tensors_num.find(input_tensor->name()) != activation_tensors_num.end())) {
                switch (Tensor::tensor_status) {
                case TENSOR_STATIC_INIT: {
                    activation_tensors_num[input_tensor->name()] += 1;
                    break;
                }
                case TENSOR_STATIC_READY: {
                    activation_tensors_num[input_tensor->name()] -= 1;
                    break;
                }
                default: {
                }
                }
                if (activation_tensors_num[input_tensor->name()] == 0 && activation_tensors[input_tensor->name()]->sequence() > 1
                    && activation_tensors[input_tensor->name()]->ttype() != GRAPH_OUTPUT) {
                    activation_tensors[input_tensor->name()]->free();
                    // std::cout << input_tensor->name() << "|" << std::endl;
                }
            }
        }
    }
#ifdef DEBUGOPTIME
    if (Tensor::tensor_status == TENSOR_STATIC_READY) {
        auto end_t = mllm_time_us();
        std::cout << op_->name() << " | " << Tensor::tensor_status << " time: " << (end_t - start_t) / 1000.0F << "ms" << std::endl;
    }
#endif
    vector<Tensor> output_result = {};
    for (const auto &layer_next_name : layer_next_names) {
        string next_name = use_layername_2_tensorname ? layername_2_tensorname[layer_next_name] : layer_next_name;
#ifdef DEBUGSAVETENSOR
        activation_tensors[next_name]->saveNData<float>(layer_next_name);
#endif
        output_result.push_back(*activation_tensors[next_name]);
    }
    return output_result;
}
*/
}; // namespace mllm