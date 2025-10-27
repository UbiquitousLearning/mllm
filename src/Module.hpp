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
#include <cassert>
#ifdef USE_OPENCL
#include "backends/opencl/OpenCLBackend.hpp"
#endif
#include <any>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory/SystemMemoryManager.hpp>
#include <memory/MemoryPoolManager.hpp>
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
} // namespace utils

class Module {
protected:
    std::shared_ptr<LlmTextGenerator> text_generator_ = nullptr;

public:
    double load_time_;
    int prefilling_token_size_ = 0;
    int decoding_token_size_ = 0;
    vector<double> inference_times_;
    BackendType device_ = BackendType::MLLM_CPU;

    map<string, shared_ptr<Tensor>> activation_tensors;
    map<string, int> activation_tensors_num;
    std::shared_ptr<AbstructLoader> loader;
    bool doLoad = false;
    bool doChangeBn = false;
    bool doTrace = false;
    bool tracedFlag = false;
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
    static bool alloc_mmap;

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

    double forwardNoInput() {
        mllm_time_init();
        vector<Tensor> tmps;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor t(Backend::global_backends[MLLM_CPU].get());
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
        return (time_end - time_start) / 1000.0F; // ms
    }

public:
    Module() {
    }
    virtual ~Module() = default;

    BackendType &device() {
        return device_;
    }

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
        if (Backend::global_backends.find(type) == Backend::global_backends.end() || Backend::global_backends[type] == nullptr) {
            // std::cout << "Initializing OpenswwssCL Backend..." << std::endl;
            // #ifdef USE_OPENCL
            //             std::cout << "Initializiwwng OpenswwssCL Backend..." << std::endl;
            // #endif

            switch (type) {
            case BackendType::MLLM_CPU: {
                shared_ptr<MemoryManager> mm = nullptr;
                // mm = std::make_shared<SystemMemoryManager>();
                mm = std::make_shared<MemoryPoolManager>(); // todomm
                Backend::global_backends[MLLM_CPU] = std::make_unique<CPUBackend>(mm);
                break;
            }
#ifdef USE_OPENCL
            case BackendType::MLLM_OPENCL: {
                // std::cout << "Initializing OpensssCL Backend..." << std::endl;
                BackendConfig config;
                Backend::global_backends[MLLM_OPENCL] = std::make_unique<OpenCLBackend>(config);
                break;
            }
#endif
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

    // TODO: Deprecated, the module is not backend specific, the backend should be set in the SubGraphStart and SubGraphFinalize
    Module &to(BackendType type) {
        initBackend(type);
        device_ = type;
        doChangeBn = true;
        doTrace = true;
        forwardNoInput();
        doChangeBn = false;
        doTrace = false;
        tracedFlag = true;
        return *this;
    }
    Module &cpu() {
        return to(MLLM_CPU);
    }
    Module &cl() {
#ifdef USE_OPENCL
        return to(MLLM_OPENCL);
#else
        throw std::runtime_error("OpenCL backend is not available. Please compile with USE_OPENCL=ON.");
#endif
    }

    void load(string path) {
        // create global loader and save to llm_model_ptr.loader as QNNBackend needs to load weights in runtime
        loader = std::make_unique<ParamLoader>(std::move(path), alloc_mmap); // todo
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        doLoad = true;
        doTrace = true;
        load_time_ = forwardNoInput(); // ms
        doLoad = false;
        tracedFlag = true;
        doTrace = false;
    }
    void load_multifile(const std::initializer_list<string> path) {
        loader = std::make_unique<MultiFileParamLoader>(std::move(path));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        doLoad = true;
        doTrace = true;
        load_time_ = forwardNoInput(); // ms
        doLoad = false;
        tracedFlag = true;
        doTrace = false;
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) = 0;

    static int graphIdx;
    string getUniqueName() {
        std::ostringstream oss;
        oss << "Module@" << idx;
        graphIdx++;
        return oss.str();
    };

    template <typename... Args>
    vector<Tensor> operator()(vector<Tensor> inputs, Args... args) {
        vector<std::any> anyArgs = convertArgsToAnyVector(args...);
        device_ = Module::llm_model_ptr->device();
        auto backend = Backend::global_backends[device_].get();
        if (inputs.empty()) {
            for (auto input : inputs) {
                assert(input.backend() == backend && "All inputs must have the same backend as the module.");
            }
        }
        if (Backend::global_backends.size() == 2 && Backend::global_backends.find(MLLM_QNN) != Backend::global_backends.end()) {
            backend = Backend::global_backends[MLLM_QNN].get();
        }
        return backend->runForward(this, inputs, anyArgs);
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

    // vector<unsigned> generate(Tensor &input_ids, const LlmTextGeneratorOpts &opt, int end_token = -1);

    vector<vector<unsigned>> generate(Tensor &input_ids, const LlmTextGeneratorOpts &opt, int end_token = -1);
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
                           vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs, vector<float> args) {
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

#define CHAINABLE_MODULE_METHODS(ClassName) \
    ClassName &to(BackendType type) {       \
        Module::to(type);                   \
        return *this;                       \
    }                                       \
    ClassName &cpu() {                      \
        to(MLLM_CPU);                       \
        return *this;                       \
    }                                       \
    ClassName &cl() {                       \
        to(MLLM_OPENCL);                    \
        return *this;                       \
    }

} // namespace mllm

#endif // MODULE_HPP
