//
// Created by Rongjie Yi on 2024/1/29 0029.
//

#ifndef OPERATION_H
#define OPERATION_H

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "Timing.hpp"

#include <Module.hpp>

#include <regex>
#include <string>
#include <vector>

namespace mllm {

class Layer {
public:
    Layer() = default;
    void init(std::string name, OpType type) {
        name_ = std::move(name);
        param_["type"] = type;
        Module::initBackend(MLLM_CPU);
        backend_ = Module::backends[MLLM_CPU];
        saved_list_idx = Module::listIdx;
        init_ = true;
    }
    bool ready() {
        return init_;
    }
    bool inited_loaded = false;
    static map<string, string> layername_2_tensorname;

    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }

    Tensor &operator()(Tensor &input0, Tensor &input1) {
        return _2I1O_OP(input0, input1);
    }

    Tensor &operator()(Tensor &input0, Tensor &input1, Tensor &input2) {
        return _3I1O_OP(input0, input1, input2);
    }

private:
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
    void reset_KVCache(string input_name) {
        vector<string> renameX_names;
        renameX_names.push_back(input_name);
        const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2","-cat", "-split-0-48"};
        for (const auto x_name : renameX_names) {
            for (auto child : Tensor::graphs[x_name]->childTensors()) {
                if (std::find(renameX_names.begin(), renameX_names.end(), child->name()) == renameX_names.end()) {
                    renameX_names.push_back(child->name());
                }
            }
        }
        for (const auto in_x_name : renameX_names) {
            for (auto suffix : suffixs) {
                if (in_x_name.rfind(suffix) == (in_x_name.size() - suffix.size())) {
                    const auto r_name = in_x_name.substr(0, in_x_name.size() - suffix.size());
                    if (std::find(renameX_names.begin(), renameX_names.end(), r_name) == renameX_names.end()) {
                        renameX_names.push_back(r_name);
                    }
                    break;
                }
            }
        }
        for (const auto x_name : renameX_names) {
            auto name = name_X_to_num(x_name, saved_list_idx);
            vector<int> shape = {Tensor::graphs[x_name]->batch(), Tensor::graphs[x_name]->head(), 
                                 Tensor::graphs[x_name]->sequence(), Tensor::graphs[x_name]->dimension()};
            layername_2_tensorname[name] = name;
            Tensor::graphs[name] = std::make_shared<Tensor>(backend_);
            Tensor::graphs[name]->initFrom(*Tensor::graphs[x_name]);
            Tensor::graphs[name]->setName(name);
            vector<Tensor *> new_chd_tensors = {};
            // for (auto child : Tensor::graphs[x_name]->childTensors()) {
            //     std::cout<<name_X_to_num(child->name(), saved_list_idx)<<std::endl;
            //     new_chd_tensors.push_back(Tensor::graphs[name_X_to_num(child->name(), saved_list_idx)].get());
            // }
            Tensor::graphs[name]->childTensors().clear();
            Tensor::graphs[name]->childTensors() = new_chd_tensors;
            if (Tensor::graphs[x_name]->aggregated() == true) {
                vector<shared_ptr<Tensor>> new_aggregated_tensors = {};
                for (const auto &aggregated_tensor : Tensor::graphs[x_name]->aggregated_tensors()) {
                    auto tmp_name = name_X_to_num(aggregated_tensor->name(), saved_list_idx);
                    if(layername_2_tensorname[tmp_name] == ""){
                        layername_2_tensorname[tmp_name] = tmp_name;
                    }
                    new_aggregated_tensors.push_back(
                        Tensor::graphs[layername_2_tensorname[name_X_to_num(aggregated_tensor->name(), saved_list_idx)]]);
                }
                Tensor::graphs[name]->addTensors(new_aggregated_tensors, Tensor::graphs[x_name]->aggregated_dim());
            }
            if(Tensor::graphs[x_name]->masterTensor() != nullptr){
                Tensor::graphs[name]->deepCopyFrom(
                    *Tensor::graphs[name_X_to_num(Tensor::graphs[x_name]->masterTensor()->name(), saved_list_idx)], 
                    false,Tensor::graphs[x_name]->shape_offset()); // b,h,s,d
            }
        }
    }
    void init_reset_KVCache(string input_name) {
        vector<string> renameX_names;
        renameX_names.push_back(input_name);
        const vector<string> suffixs = {"-view", ".split-0", ".split-1", ".split-2","-cat", "-split-0-48"};
        for (const auto in_x_name : renameX_names) {
            for (auto suffix : suffixs) {
                if (in_x_name.rfind(suffix) == (in_x_name.size() - suffix.size())) {
                    const auto r_name = in_x_name.substr(0, in_x_name.size() - suffix.size());
                    if (std::find(renameX_names.begin(), renameX_names.end(), r_name) == renameX_names.end()) {
                        renameX_names.push_back(r_name);
                    }
                    break;
                }
            }
        }
        for (const auto x_name : renameX_names) {
            auto name = name_X_to_num(x_name, saved_list_idx);
            layername_2_tensorname[name] = name;
            Tensor::graphs[name] = std::make_shared<Tensor>(backend_);
            Tensor::graphs[name]->initFrom(*Tensor::graphs[x_name]);
            Tensor::graphs[name]->setName(name);
        }
    }

protected:
    bool INIT_OP() {
        if (op_ == nullptr) {
            op_ = backend_->opCreate(param_, name_);
        }
        if (Module::doLoad) {
            op_->load(*Module::loader);
            inited_loaded=true;
        } else {
            if(!inited_loaded){
                Module::loader= new ParamLoader("");
                op_->load(*Module::loader);
                inited_loaded=true;
            }
        }
        return Module::doLoad;
    }
    Tensor &_1I1O_OP(Tensor &input) {
        Module::runlistIdx = saved_list_idx;
        if (Module::doLoad || !inited_loaded) {
            INIT_OP();
            string layer_next_name = "out-" + op_->name();
            if (Tensor::graphs.find(input.name()) == Tensor::graphs.end() || input.count() != Tensor::graphs[input.name()]->count()) {
                Tensor::graphs[input.name()] = std::shared_ptr<Tensor>(&input, [](Tensor *) {});
                Tensor::graphs[input.name()]->setName(input.name());
            } 
            auto in_name = input.name();
            if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                if (param_["type"] == KVCACHE) {
                    layername_2_tensorname[layer_next_name] = layer_next_name;
                    init_reset_KVCache(input.name());
                    in_name = name_X_to_num(in_name, saved_list_idx);
                } else {
                    layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                }
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
                Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
                Tensor::graphs[next_name]->setName(next_name);
            }
            if(Module::doLoad){
                return *Tensor::graphs[next_name];
            }
        }
        string layer_next_name = "out-" + op_->name();
        auto next_name = layername_2_tensorname[layer_next_name];
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
            case TENSOR_STATIC_INIT: {
                op_->reshape({Tensor::graphs[input.name()]}, {Tensor::graphs[next_name]});
                op_->setUp({Tensor::graphs[input.name()]}, {Tensor::graphs[next_name]});
                break;
            }
            case TENSOR_STATIC_READY: {
                op_->execute({Tensor::graphs[input.name()]}, {Tensor::graphs[next_name]});
                break;
            }
            default: {
                break;
            }
        }
#ifdef DEBUGOPTIME
        auto end_t = mllm_time_us();
        std::cout<<op_->name() << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
        Tensor::graphs[next_name]->saveNData<float>(layer_next_name);
#endif     
        return *Tensor::graphs[next_name];        
    }
    Tensor &_2I1O_OP(Tensor &input0, Tensor &input1) {
        Module::runlistIdx = saved_list_idx;
        if (Module::doLoad || !inited_loaded) {
            INIT_OP();
            string layer_next_name = "out-" + op_->name();
            if (Tensor::graphs.find(input0.name()) == Tensor::graphs.end() || input0.count() != Tensor::graphs[input0.name()]->count()) {
                Tensor::graphs[input0.name()] = std::shared_ptr<Tensor>(&input0, [](Tensor *) {});
                Tensor::graphs[input0.name()]->setName(input0.name());
            }
            if (Tensor::graphs.find(input1.name()) == Tensor::graphs.end() || input1.count() != Tensor::graphs[input1.name()]->count()) {
                Tensor::graphs[input1.name()] = std::shared_ptr<Tensor>(&input1, [](Tensor *) {});
                Tensor::graphs[input1.name()]->setName(input1.name());
            }
            if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
                Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
                Tensor::graphs[next_name]->setName(next_name);
            }
            if(Module::doLoad){
                return *Tensor::graphs[next_name];
            }
        }
        string layer_next_name = "out-" + op_->name();
        auto next_name = layername_2_tensorname[layer_next_name];
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
            case TENSOR_STATIC_INIT: {
                op_->reshape({Tensor::graphs[input0.name()], Tensor::graphs[input1.name()]}, {Tensor::graphs[next_name]});
                op_->setUp({Tensor::graphs[input0.name()], Tensor::graphs[input1.name()]}, {Tensor::graphs[next_name]});
                break;
            }
            case TENSOR_STATIC_READY: {
                op_->execute({Tensor::graphs[input0.name()], Tensor::graphs[input1.name()]}, {Tensor::graphs[next_name]});
                break;
            }
            default: {
                break;
            }
        }
#ifdef DEBUGOPTIME
        auto end_t = mllm_time_us();
        std::cout<<op_->name() << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
        Tensor::graphs[next_name]->saveNData<float>(layer_next_name);
#endif
        return *Tensor::graphs[next_name];
    }
    Tensor &_3I1O_OP(Tensor &input0, Tensor &input1, Tensor &input2) {
        Module::runlistIdx = saved_list_idx;
        if (Module::doLoad || !inited_loaded) {
            INIT_OP();
            string layer_next_name = "out-" + op_->name();   
            if (Tensor::graphs.find(input0.name()) == Tensor::graphs.end() || input0.count() != Tensor::graphs[input0.name()]->count()) {
                Tensor::graphs[input0.name()] = std::shared_ptr<Tensor>(&input0, [](Tensor *) {});
                Tensor::graphs[input0.name()]->setName(input0.name());
            }
            if (Tensor::graphs.find(input1.name()) == Tensor::graphs.end() || input1.count() != Tensor::graphs[input1.name()]->count()) {
                Tensor::graphs[input1.name()] = std::shared_ptr<Tensor>(&input1, [](Tensor *) {});
                Tensor::graphs[input1.name()]->setName(input1.name());
            }
            if (Tensor::graphs.find(input2.name()) == Tensor::graphs.end() || input2.count() != Tensor::graphs[input2.name()]->count()) {
                Tensor::graphs[input2.name()] = std::shared_ptr<Tensor>(&input2, [](Tensor *) {});
                Tensor::graphs[input2.name()]->setName(input2.name());
            }
            if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
                Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
                Tensor::graphs[next_name]->setName(next_name);
            }
            if(Module::doLoad){
                return *Tensor::graphs[next_name];
            }
        }
        string layer_next_name = "out-" + op_->name();
        auto next_name = layername_2_tensorname[layer_next_name];
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
            case TENSOR_STATIC_INIT: {
                op_->reshape({Tensor::graphs[input0.name()], Tensor::graphs[input1.name()], Tensor::graphs[input2.name()]}, 
                            {Tensor::graphs[next_name]});
                op_->setUp({Tensor::graphs[input0.name()], Tensor::graphs[input1.name()], Tensor::graphs[input2.name()]}, 
                            {Tensor::graphs[next_name]});
                break;
            }
            case TENSOR_STATIC_READY: {
                op_->execute({Tensor::graphs[input0.name()], Tensor::graphs[input1.name()], Tensor::graphs[input2.name()]}, 
                            {Tensor::graphs[next_name]});
                break;
            }
            default: {
                break;
            }
        }
#ifdef DEBUGOPTIME
        auto end_t = mllm_time_us();
        std::cout<<op_->name() << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
        Tensor::graphs[next_name]->saveNData<float>(layer_next_name);
#endif
        return *Tensor::graphs[next_name];
    }
    Tensor &_3I1OO1_OP(Tensor &input0, Tensor &input1, Tensor &input2) {
        Module::runlistIdx = saved_list_idx;
        if (Module::doLoad || !inited_loaded) {
            INIT_OP();
            string layer_next_name = "out-" + op_->name();
            if (Tensor::graphs.find(input0.name()) == Tensor::graphs.end() || input0.count() != Tensor::graphs[input0.name()]->count()) {
                Tensor::graphs[input0.name()] = std::shared_ptr<Tensor>(&input0, [](Tensor *) {});
                Tensor::graphs[input0.name()]->setName(input0.name());
            }
            if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
                Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
                Tensor::graphs[next_name]->setName(next_name);
            }
            if(Module::doLoad){
                return *Tensor::graphs[next_name];
            }
        }
        string layer_next_name = "out-" + op_->name();
        auto next_name = layername_2_tensorname[layer_next_name];
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
            case TENSOR_STATIC_INIT: {
                op_->reshape({Tensor::graphs[input0.name()], 
                                std::shared_ptr<Tensor>(&input1, [](Tensor *) {}), 
                                std::shared_ptr<Tensor>(&input2, [](Tensor *) {})}, 
                            {Tensor::graphs[next_name]});
                op_->setUp({Tensor::graphs[input0.name()], 
                                std::shared_ptr<Tensor>(&input1, [](Tensor *) {}), 
                                std::shared_ptr<Tensor>(&input2, [](Tensor *) {})}, 
                            {Tensor::graphs[next_name]});
                break;
            }
            case TENSOR_STATIC_READY: {
                op_->execute({Tensor::graphs[input0.name()], 
                                std::shared_ptr<Tensor>(&input1, [](Tensor *) {}), 
                                std::shared_ptr<Tensor>(&input2, [](Tensor *) {})}, 
                            {Tensor::graphs[next_name]});
                break;
            }
            default: {
                break;
            }
        }
#ifdef DEBUGOPTIME
        auto end_t = mllm_time_us();
        std::cout<<op_->name() << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
        Tensor::graphs[next_name]->saveNData<float>(layer_next_name);
 #endif
        return *Tensor::graphs[next_name];
    }
    Tensor &_0I1O_OP() {
        Module::runlistIdx = saved_list_idx;
        if (Module::doLoad || !inited_loaded) {
            INIT_OP();
            string layer_next_name = "param-" + op_->name();
            if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
            }
            auto next_name = layername_2_tensorname[layer_next_name];
            if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
                Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
                Tensor::graphs[next_name]->setName(next_name);
            }
            if(Module::doLoad){
                return *Tensor::graphs[next_name];
            }
        }
        string layer_next_name = "param-" + op_->name();
        auto next_name = layername_2_tensorname[layer_next_name];
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
            case TENSOR_STATIC_INIT: {
                op_->reshape({}, {Tensor::graphs[next_name]});
                op_->setUp({}, {Tensor::graphs[next_name]});
                break;
            }
            case TENSOR_STATIC_READY: {
                op_->execute({}, {Tensor::graphs[next_name]});
                break;
            }
            default: {
                break;
            }
        }
#ifdef DEBUGOPTIME
        auto end_t = mllm_time_us();
        std::cout<<op_->name() << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
#ifdef DEBUGSAVETENSOR
        Tensor::graphs[next_name]->saveNData<float>(layer_next_name);
#endif
        return *Tensor::graphs[next_name];
    }
    vector<Tensor> _1INO_OP(Tensor &input, int N) {
        Module::runlistIdx = saved_list_idx;
        if (Module::doLoad || !inited_loaded) {
            INIT_OP();
            vector<string> layer_next_names = {};
            for (int i = 0; i < N; ++i) {
                layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
            }
            if (Tensor::graphs.find(input.name()) == Tensor::graphs.end()) {
                Tensor::graphs[input.name()] = std::shared_ptr<Tensor>(&input, [](Tensor *) {});
                Tensor::graphs[input.name()]->setName(input.name());
            } 
            for (const auto &layer_next_name : layer_next_names) {
                if (layername_2_tensorname.find(layer_next_name) == layername_2_tensorname.end()) {
                    layername_2_tensorname[layer_next_name] = name_num_to_X(layer_next_name);
                }
                auto next_name = layername_2_tensorname[layer_next_name];
                if (Tensor::graphs.find(next_name) == Tensor::graphs.end()) {
                    Tensor::graphs[next_name] = std::make_shared<Tensor>(backend_);
                    Tensor::graphs[next_name]->setName(next_name);
                }
            }
            if(Module::doLoad){
                vector<Tensor> output_result = {};
                for (const auto &layer_next_name : layer_next_names) {
                    auto next_name = layername_2_tensorname[layer_next_name];
                    output_result.push_back(*Tensor::graphs[next_name]);
                }
                return output_result;
            }
        }
        vector<string> layer_next_names = {};
        for (int i = 0; i < N; ++i) {
            layer_next_names.push_back("out-" + op_->name() + "-" + std::to_string(i));
        }
        vector<shared_ptr<Tensor>> shared_outputs = {};
        vector<string> next_names = {};
        for (const auto &layer_next_name : layer_next_names) {
            auto next_name = layername_2_tensorname[layer_next_name];
            next_names.push_back(next_name);
            shared_outputs.push_back(Tensor::graphs[next_name]);
        }
#ifdef DEBUGOPTIME
        auto start_t = mllm_time_us();
#endif
        switch (Tensor::tensor_status) {
            case TENSOR_STATIC_INIT: {
                op_->reshape({ Tensor::graphs[input.name()]}, shared_outputs);
                op_->setUp({ Tensor::graphs[input.name()]}, shared_outputs);
                break;
            }
            case TENSOR_STATIC_READY: {
                op_->execute({ Tensor::graphs[input.name()]}, shared_outputs);
                break;
            }
            default: {
                break;
            }
        }
#ifdef DEBUGOPTIME
        auto end_t = mllm_time_us();
        std::cout<<op_->name() << " | "<<Tensor::tensor_status<<" time: " << (end_t - start_t)/1000.0F <<"ms"<< std::endl;
#endif
        vector<Tensor> output_result = {};
        for (const auto &layer_next_name : layer_next_names) {
            auto next_name = layername_2_tensorname[layer_next_name];
#ifdef DEBUGSAVETENSOR
            Tensor::graphs[next_name]->saveNData<float>(layer_next_name);
#endif
            output_result.push_back(*Tensor::graphs[next_name]);
        }
        return output_result;
    }

    std::string name_;
    Op *op_ = nullptr;
    Backend *backend_{};
    OpParam param_;
    bool init_ = false;
    int saved_list_idx;
};

class Linear final : public Layer {
public:
    explicit Linear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LINEAR);
    }
    Tensor &operator()(Tensor &input){
        return _1I1O_OP(input);
    }
};

class SparseIdLinear final : public Layer{
public:
    SparseIdLinear(int in_dim, int out_dim, std::string name){
        param_["in_dim_"] = (float) in_dim;
        param_["out_dim_"] = (float) out_dim;
        init(std::move(name), OpType::SPARSEIDLINEAR);
    }
    Tensor &operator()(Tensor &input) {
            return _1I1O_OP(input);
    }
};

class SparseLinear final : public Layer{
public:
    SparseLinear(int in_dim, int out_dim, std::string name){
        param_["in_dim_"] = (float) in_dim;
        param_["out_dim_"] = (float) out_dim;
        init(std::move(name), OpType::SPARSELINEAR);
    }
    Tensor &operator()(Tensor &input) {
            return _1I1O_OP(input);
    }
};

class Predictor final : public Layer {
public:
    Predictor(int in_dim, int out_dim, std::string name){
        param_["in_dim"] = (float) in_dim;
        param_["out_dim"] = (float) out_dim;
        init(std::move(name), OpType::PREDICTOR);
    }
    Tensor &operator()(Tensor &input) {
            return _1I1O_OP(input);
    }
};

class ElasticLinear final : public Layer {
public:
    ElasticLinear() = default;
    explicit ElasticLinear(int in_features, int out_features, bool bias, std::string name) {
        param_["in_features"] = in_features;
        param_["out_features"] = out_features;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::ELASTICLINEAR);
    }
    Tensor &operator()(Tensor &input0, int activate_input_dim, int activate_output_dim) {
        auto activate_input_dim_tensor = Tensor(1, 1, 1, 1, backend_, true);
        activate_input_dim_tensor.setDataAt<float>(0,0,0,0,(float)activate_input_dim);
        auto activate_output_dim_tensor = Tensor(1, 1, 1, 1, backend_, true);
        activate_output_dim_tensor.setDataAt<float>(0,0,0,0,(float)activate_output_dim);
        return _3I1OO1_OP(input0, activate_input_dim_tensor, activate_output_dim_tensor);
    }
};


class SiLU final : public Layer {
public:
    SiLU() = default;
    SiLU(std::string name) {
        init(std::move(name), OpType::SILU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class ReLU final : public Layer {
public:
    ReLU() = default;
    ReLU(std::string name) {
        init(std::move(name), OpType::RELU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class ReLUSquaredActivation final : public Layer {
public:
    ReLUSquaredActivation() = default;
    ReLUSquaredActivation(std::string name) {
        init(std::move(name), OpType::RELU2);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class GELU final : public Layer {
public:
    GELU() = default;
    GELU(std::string name) {
        init(std::move(name), OpType::OP_GELU);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class QuickGELU final : public Layer {
public:
    QuickGELU() = default;
    explicit QuickGELU(std::string name) {
        init(std::move(name), OpType::QUICKGLUE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

using ActFnConstructor = std::function<Layer(const std::string &)>;
inline std::map<std::string, ActFnConstructor> ACT_FN = {
    {"SiLU", [](const std::string &name) { return SiLU(name); }},
    {"ReLU", [](const std::string &name) { return ReLU(name); }},
    {"ReLU2", [](const std::string &name) { return ReLUSquaredActivation(name); }},
    {"GELU", [](const std::string &name) { return GELU(name); }},
    {"QuickGELU", [](const std::string &name) { return QuickGELU(name); }},
};

class Softmax final : public Layer {
public:
    Softmax() = default;
    explicit Softmax(Chl axis, std::string name) {
        param_["axis"] = axis;
        init(std::move(name), OpType::SOFTMAX);
    }
    explicit Softmax(Chl axis, bool do_causal_mask, std::string name) {
        param_["axis"] = axis;
        param_["do_causal_mask"] = do_causal_mask;
        init(std::move(name), OpType::SOFTMAX);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
    Tensor &operator()(Tensor &input, int axis_classes) {
        auto axis_classes_tensor = Tensor(1, 1, 1, 1, backend_, true);
        axis_classes_tensor.setDataAt<float>(0,0,0,0,(float)axis_classes);
        return _3I1OO1_OP(input, axis_classes_tensor, axis_classes_tensor);
    }
};

class Embedding final : public Layer {
public:
    explicit Embedding(int vocab_size, int hidden_size, std::string name) {
        param_["hidden_size"] = hidden_size;
        param_["vocab_size"] = vocab_size;
        init(std::move(name), OpType::EMBEDDING);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Causalmask final : public Layer {
public:
    Causalmask() = default;
    explicit Causalmask(std::string name) {
        init(std::move(name), OpType::CAUSALMASK);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
    Tensor &operator()(Tensor &input0, int kvcache_seq) {
        auto kvcache_seq_tensor = Tensor(1, 1, 1, 1, backend_, true);
        kvcache_seq_tensor.setDataAt<float>(0,0,0,0,(float)kvcache_seq);
        return _3I1OO1_OP(input0, kvcache_seq_tensor, kvcache_seq_tensor);
    }
};

class SlidingWindowMask final : public Layer {
public:
    explicit SlidingWindowMask(int window_size, std::string name) {
        param_["window_size"] = window_size;
        init(std::move(name), OpType::SLIDINGWINDOWMASK);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class RoPE final : public Layer {
public:
    explicit RoPE(int pose_type, std::string name) {
        param_["pose_type"] = pose_type;
        init(std::move(name), OpType::ROPE);
    }
    explicit RoPE(int pose_type, float rope_theta, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        init(std::move(name), OpType::ROPE);
    }
    explicit RoPE(int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings, std::string name) {
        param_["pose_type"] = pose_type;
        param_["rope_theta"] = rope_theta;
        param_["max_position_embeddings"] = max_position_embeddings;
        param_["partial_rotary_factor"] = partial_rotary_factor;
        init(std::move(name), OpType::ROPE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class KVCache final : public Layer {
public:
    KVCache() = default;
    explicit KVCache(int cache_max, std::string name) {
        param_["n_rep"] = 1;
        param_["cache_max"] = cache_max;
        init(std::move(name), OpType::KVCACHE);
    }
    explicit KVCache(int n_rep, int cache_max, std::string name) {
        param_["n_rep"] = n_rep;
        param_["cache_max"] = cache_max;
        init(std::move(name), OpType::KVCACHE);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
    int getCacheSeqLen(){
        return op_->getCacheSeqLen();
    }
    void clearCache(){
        return op_->clearCache();
    }
};

class LayerNorm final : public Layer {
public:
    explicit LayerNorm(int norm_size, bool bias, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::LAYERNORM);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class RMSNorm final : public Layer {
public:
    explicit RMSNorm(int norm_size, float epsilon, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        init(std::move(name), OpType::RMSNORM);
    }

    explicit RMSNorm(int norm_size, float epsilon, bool add_unit_offset, std::string name) {
        param_["norm_size"] = norm_size;
        param_["epsilon"] = epsilon;
        param_["add_unit_offset"] = (float)add_unit_offset;
        init(std::move(name), OpType::RMSNORM);
    }

    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Matmul final : public Layer {
public:
    explicit Matmul(bool transpose0, bool transpose1, std::string name) {
        param_["transpose0"] = transpose0;
        param_["transpose1"] = transpose1;
        init(std::move(name), OpType::MATMUL);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        return _2I1O_OP(input0, input1);
    }
};

class Split final : public Layer {
public:
    Split() = default;

    explicit Split(int split_num, Chl split_dim, int split_dim_size, std::string name) {
        param_["split_num"] = (float)split_num;
        param_["split_dim"] = (float)split_dim;
        param_["split_dim_size"] = (float)split_dim_size;
        init(std::move(name), OpType::SPLIT);
    }

    explicit Split(const std::vector<int> &each_dims, Chl split_dim, const std::string &name) {
        param_["split_num"] = (float)each_dims.size();
        param_["split_dim"] = (float)split_dim;
        // store each dims
        for (size_t i = 0; i < each_dims.size(); ++i) {
            param_["split_dim_size_" + std::to_string(i)] = (float)each_dims[i];
        }
        init(std::move(name), OpType::SPLIT);
    }

    vector<Tensor> operator()(Tensor &input) {
        return _1INO_OP(input, (int)param_["split_num"]);
    }
};

class Convolution2D final : public Layer {
public:
    explicit Convolution2D(int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, std::string name) {
        param_["in_channel"] = (float)in_channel;
        param_["out_channel"] = (float)out_channel;
        param_["kernal_h"] = (float)kernal[0];
        param_["kernal_w"] = (float)kernal[1];
        param_["stride_h"] = (float)stride[0];
        param_["stride_w"] = (float)stride[1];
        param_["padding"] = (float)padding;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::CONVOLUTION2D);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Convolution3D final : public Layer {
public:
    explicit Convolution3D(int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias, std::string name) {
        param_["in_channel"] = (float)in_channel;
        param_["out_channel"] = (float)out_channel;
        param_["kernal_t"] = (float)kernal[0];
        param_["kernal_h"] = (float)kernal[1];
        param_["kernal_w"] = (float)kernal[2];
        param_["stride_t"] = (float)stride[0];
        param_["stride_h"] = (float)stride[1];
        param_["stride_w"] = (float)stride[2];
        param_["padding"] = (float)padding;
        param_["bias"] = (float)bias;
        init(std::move(name), OpType::CONVOLUTION3D);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

class Concat final : public Layer {
public:
    explicit Concat(Chl axis, std::string name) {
        param_["axis"] = (float)axis;
        init(std::move(name), OpType::CAT);
    }
    Tensor &operator()(Tensor &input0, Tensor &input1) {
        return _2I1O_OP(input0, input1);
    }
};

class Parameter final : public Layer {
public:
    Parameter() = default;
    explicit Parameter(int batch, int seq, int head, int dim, std::string name) {
        param_["batch"] = batch;
        param_["seq"] = seq;
        param_["head"] = head;
        param_["dim"] = dim;
        init(std::move(name), OpType::PARAMETER);
    }
    Tensor &operator()() {
        return _0I1O_OP();
    }
};

class Position final : public Layer {
public:
    explicit Position(std::string name) {
        init(std::move(name), OpType::POSITION);
    }
    Tensor &operator()(Tensor &input) {
        return _1I1O_OP(input);
    }
};

} // namespace mllm

#endif // OPERATION_H
