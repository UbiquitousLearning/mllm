//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#ifndef MODULE_HPP
#define MODULE_HPP
#include "Generate.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "Timing.hpp"
#include "Trace.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"
#include <any>
#include <functional>
#include <iostream>
#include <memory/SystemMemoryManager.hpp>
#include <memory>
#include <ostream>
#include <stack>
#include <utility>
#include <vector>
#include <unordered_map>

namespace mllm {

namespace utils {
// get the closest factors of a number, used in NPU part2 view to speed up the QNN linear
inline std::pair<int, int> closestFactors(int n) {
    int root = static_cast<int>(sqrt(n));
    for (int i = root; i > 0; --i) {
        if (n % i == 0) {
            return {i, n / i};
        }
    }
    return {1, n};
}
}

class Module {
protected:
    double load_time_;
    int prefilling_token_size_ = 0;
    int decoding_token_size_ = 0;
    vector<double> inference_times_;
    vector<vector<int>> last_shape_bshd_;
    std::shared_ptr<LlmTextGenerator> text_generator_ = nullptr;
    BackendType device_ = BackendType::MLLM_CPU;

public:
    map<string, shared_ptr<Tensor>> activation_tensors;
    map<string, int> activation_tensors_num;
    AbstructLoader *loader;
    bool doLoad = false;
    bool op_transposed_flag = false;

    static Module *llm_model_ptr;
    // tag to indicate the multi-chunk prefilling
    static bool isMultiChunkPrefilling;
    // tag to indicate the first chunk
    static bool isFirstChunk;

    static int listIdx;
    static std::stack<int> listIdxStack;
    // static int runlistIdx;

    static bool doToDevice;
    static BackendType tmp_device;

    static std::unordered_map<string, shared_ptr<Op>> tensor_func_ops; // use for QNN

private:
    template <typename... Args>
    vector<std::any> convertArgsToAnyVector(Args... args) {
        return vector<std::any>{std::any(args)...};
    }

    // 递归终止函数
    template <typename T>
    static auto change_last(T value) {
        return std::make_tuple(value + std::to_string(listIdx) + ".");
    }
    // 递归函数
    template <typename T, typename... Args>
    static auto change_last(T head, Args... tail) {
        auto tail_tuple = change_last(tail...);
        return std::tuple_cat(std::make_tuple(head), tail_tuple);
    }
    int idx;
public:
    Module() {
        idx = Module::graphIdx;
        Module::graphIdx++;
    }
    virtual ~Module() = default;

    BackendType device() const {
        return device_;
    }

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
        if (Backend::global_backends.find(type) == Backend::global_backends.end() || Backend::global_backends[type] == nullptr) {
            switch (type) {
            case BackendType::MLLM_CPU: {
                shared_ptr<MemoryManager> mm = nullptr;
                mm = std::make_shared<SystemMemoryManager>();
                Backend::global_backends[MLLM_CPU] = new CPUBackend(mm);
                break;
            }
#ifdef USE_QNN
            case BackendType::MLLM_QNN: {
                Backend::global_backends.emplace(MLLM_QNN, GetBackendCreator(MLLM_QNN)->create({}));
                break;
            }
#endif
#ifdef MLLM_BUILD_XNNPACK_BACKEND
            case BackendType::MLLM_XNNPACK: {
                Backend::global_backends.emplace(MLLM_XNNPACK, GetBackendCreator(MLLM_XNNPACK)->create({}));
                break;
            }
#endif
            default: {
            }
            }
        }
    }
    void to(BackendType type) {
        initBackend(type);
        device_ = type;
    }

    void load(string path) {
        // create global loader and save to llm_model_ptr.loader as QNNBackend needs to load weights in runtime
        loader = new ParamLoader(std::move(path));
        load(*loader);
    }
    void load(AbstructLoader &param_loader) {
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        mllm_time_init();

        loader = &param_loader;
        doLoad = true;
        vector<Tensor> tmps;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor t(Backend::global_backends[MLLM_CPU]);
            t.setName("input" + std::to_string(i));
            t.reshape(1, 1, 1, 10);
            t.alloc();
            t.setModule(this);
            tmps.push_back(t);
        }
        llm_model_ptr = this;
        vector<std::any> alternate_args = {
            {},
            vector<int>{0, 0},
            std::vector<std::vector<int>>(32, std::vector<int>(2))};
        uint64_t time_start = 0;
        for (auto args : alternate_args) {
            time_start = mllm_time_us();
            try {
                operator()(tmps, args);
                break;
            } catch (const std::exception &e) {
#if not defined(__ARM_NEON)
                if (std::string("bad any_cast") != e.what()) {
                    MLLM_LOG_ERROR_STREAM << e.what() << std::endl;
                    exit(0);
                }
#endif
            } catch (...) {
                MLLM_LOG_ERROR_STREAM << "load error" << std::endl;
                exit(0);
            }
        }
        uint64_t time_end = mllm_time_us();
        load_time_ = (time_end - time_start) / 1000.0F; // ms
        doLoad = false;
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) = 0;

    static int graphIdx;
    string getUinqueName(){
        std::ostringstream oss;
        oss << "Module@" << idx;
        graphIdx++;
        return oss.str();
    };

    template <typename... Args>
    vector<Tensor> operator()(vector<Tensor> inputs, Args... args) {
        vector<std::any> anyArgs = convertArgsToAnyVector(args...);
        // set static tmp_device to device_ to init layers' op
        auto previoud_device = tmp_device;
        Module::tmp_device = device_;
        // Module Loading
        if (llm_model_ptr && llm_model_ptr->doLoad) {
            auto outputs = Forward(inputs, anyArgs);
            // for inner module, set output tensors to GRAPH_OUTPUT
            if (inputs[0].ttype() != TensorType::INPUT_TENSOR) { // XPUs' module should not be the outermost input tensor
                for (auto &output : outputs) {
                    inputs[0].module()->activation_tensors[output.name()]->setTtype(GRAPH_OUTPUT);
                }
            }
            // set Module::tmp_device to previous device
            Module::tmp_device = previoud_device;
            return outputs;
        }
        // Module setUp & execute
        if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
            if (prefilling_token_size_ == 0) { // first time init
                prefilling_token_size_ = inputs[0].sequence();
            } else if (decoding_token_size_ == 0) {
                decoding_token_size_ = inputs[0].sequence();
            }
            for (int i = 0; i < inputs.size(); i++) {
                auto &input = inputs[i];
                input.setName("input" + std::to_string(i));
                input.setTtype(TensorType::NORMAL_TENSOR);
                activation_tensors[input.name()] = std::shared_ptr<Tensor>(&input, [](Tensor *) {});
                activation_tensors[input.name()]->setName(input.name());
                activation_tensors[input.name()]->setModule(this);
            }
            llm_model_ptr = this;
            Tensor::tensor_status = TENSOR_STATIC_INIT;

            uint64_t time_start = mllm_time_us();
            Forward(inputs, anyArgs);
            Tensor::tensor_status = TENSOR_STATIC_READY;
            // uint64_t time_start = mllm_time_us();
            auto output = Forward(inputs, anyArgs);
            uint64_t time_end = mllm_time_us();

            double inference_time_ = (time_end - time_start) / 1000.0F; // ms
            inference_times_.push_back(inference_time_);
            last_shape_bshd_.clear();
            for (auto &input : inputs) {
                last_shape_bshd_.push_back({input.batch(), input.sequence(),
                                            input.head(), input.dimension()});
            }
            llm_model_ptr->op_transposed_flag = true;
            return output;
        } else { // inner Modules
            // offload according to the backends' info inited during loading
            if (Tensor::tensor_status == TENSOR_STATIC_INIT && device_ != MLLM_CPU) { // backend specific module reshape & setup
                if (Module::isMultiChunkPrefilling && !Module::isFirstChunk) {        // set to TENSOR_UNDEFINED and SKIP executing qnn layers
                    Tensor::tensor_status = TENSOR_UNDEFINED;
                    auto outputs = Forward(inputs, anyArgs);
                    Tensor::tensor_status = TENSOR_STATIC_INIT;
                    return outputs;
                }
                auto inputs_vec = vector<shared_ptr<Tensor>>();
                auto outputs_vec = vector<shared_ptr<Tensor>>();
                for (auto &i : inputs) {
                    inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
                }

                Backend::global_backends[device_]->onSetUpStart(inputs_vec, outputs_vec, getUinqueName());

                // for xnnpack currently
                for (auto &i : inputs) {
                    i.uuid() = inputs[0].module()->activation_tensors[i.name()]->uuid();
                }

                auto outputs = Forward(inputs, anyArgs);
                for (auto &output : outputs) {
                    outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
                }
                Backend::global_backends[device_]->onSetUpEnd(inputs_vec, outputs_vec, getUinqueName());

                // for xnnpack currently
                for (auto &o : outputs) {
                    o.uuid() = outputs[0].module()->activation_tensors[o.name()]->uuid();
                }

                return outputs;
            } else if (Tensor::tensor_status == TENSOR_STATIC_READY && device_ != MLLM_CPU) { // backend specific module execute
                auto inputs_vec = vector<shared_ptr<Tensor>>();
                auto outputs_vec = vector<shared_ptr<Tensor>>();
                for (auto &i : inputs) {
                    inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
                }

                auto outputs = Forward(inputs, anyArgs);

                for (auto &output : outputs) {
                    outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
                }
                Backend::global_backends[device_]->onExecuteStart(inputs_vec, outputs_vec, getUinqueName());

                Backend::global_backends[device_]->onExecuteEnd(outputs_vec, getUinqueName());

                // for xnnpack currently
                for (auto &o : outputs) {
                    o.uuid() = outputs[0].module()->activation_tensors[o.name()]->uuid();
                    o.forceResetHostPointer(outputs[0].module()->activation_tensors[o.name()]->rawHostPtr());
                }

                return outputs;
            } else if (Tensor::tensor_status == TENSOR_STATIC_TRACE && device_ != MLLM_CPU) {
                auto inputs_vec = vector<shared_ptr<Tensor>>();
                auto outputs_vec = vector<shared_ptr<Tensor>>();
                for (auto &i : inputs) {
                    inputs_vec.push_back(inputs[0].module()->activation_tensors[i.name()]);
                }

                auto outputs = Forward(inputs, anyArgs);

                for (auto &output : outputs) {
                    outputs_vec.push_back(inputs[0].module()->activation_tensors[output.name()]);
                }
                Tracer::addModule(inputs_vec, outputs_vec, getUinqueName());
                return outputs;
            }
            return Forward(inputs, anyArgs);
        }
    }

    template <typename T, typename... Args>
    static vector<T> List(int n, Args &&...args) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        if (listIdx) {
            listIdxStack.push(listIdx);
        }
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            auto new_args = change_last(args...);
            modules.push_back(std::move(T(std::apply([&](auto &&...args) { return T(std::forward<decltype(args)>(args)...); }, new_args))));
            listIdx++;
        }
        if (!listIdxStack.empty()) {
            listIdx = listIdxStack.top();
            listIdxStack.pop();
        } else {
            listIdx = 0;
        }
        return modules;
    }

    void free() {
        activation_tensors.clear();
    }

    void setNoLoadWeightsDtype(DataType dtype) {
        llm_model_ptr = this;
        Op::noLoadWeightsDtype() = dtype;
    }
    virtual void clear_kvcache() {
        ;
    }
    vector<double> profiling(string name = "");
    virtual void generate(
        Tensor &input_ids, const LlmTextGeneratorOpts &opt, const std::function<bool(unsigned int)> &call_back = [](unsigned int) -> bool { return true; });

    vector<unsigned> generate(Tensor &input_ids, const LlmTextGeneratorOpts &opt, int end_token = -1);
};

class CPUModuleWrapper : public Module {
public:
    vector<shared_ptr<Callable>> traces_;

    void addOp(Op *op,
               vector<shared_ptr<Tensor>> inputs,
               vector<shared_ptr<Tensor>> outputs) {
        auto callable = std::make_shared<Callable>(CallableType::OP);
        callable->opInputs = inputs;
        callable->opOutputs = outputs;
        callable->op = op;
        traces_.push_back(callable);
    }

    void addTensorFunction(TensorFunction *func,
                           vector<Tensor *> inputs, vector<Tensor *> outputs, vector<float> args) {
        auto callable = std::make_shared<Callable>(CallableType::TENSOR_FUNC);
        callable->tensorFunc = func;
        callable->tensorInputs = inputs;
        callable->tensorOutputs = outputs;
        for (auto arg : args) {
            callable->args.push_back(arg);
        }
        traces_.push_back(callable);
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        // get chunk_id from args
        int chunk_id = std::any_cast<int>(args[0]);
        if (chunk_id != 0) {
            for (auto &callable : traces_) {
                callable->reshape();
                callable->setUp();
            }
        }

        for (int i = 0; i < traces_.size(); i++) {
            traces_[i]->execute();
        }
        return {};
    }

    vector<shared_ptr<Tensor>> result() {
        return traces_.back()->outputs();
    }
};

class QNNModuleWrapper : public Module {
public:
    string name_;
    vector<shared_ptr<Tensor>> inputs_;
    vector<shared_ptr<Tensor>> outputs_;

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) override {
        Backend::global_backends[MLLM_QNN]->onExecuteStart(inputs_, outputs_, name_);
        Backend::global_backends[MLLM_QNN]->onExecuteEnd(outputs_, name_);
        return {};
    }
};

} // namespace mllm

#endif // MODULE_HPP
