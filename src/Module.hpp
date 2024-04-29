//
// Created by Rongjie Yi on 2024/1/28 0028.
//

#ifndef MODULE_HPP
#define MODULE_HPP
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Backend.hpp"
#include "backends/cpu/CPUBackend.hpp"
#ifdef USE_QNN
#include "backends/qnn/QNNBackend.hpp"
#endif

#include <any>
#include <memory/SystemMemoryManager.hpp>
#include <memory>
#include <utility>

namespace mllm {

class Module {
protected:
    Backend *backend_ = backends[MLLM_CPU];
public:
    static map<BackendType, Backend *> backends;
    static ParamLoader *loader;
    static TensorStatus tensor_status;
    static bool doLoad;
    static bool doToDevice;
    static BackendType tmp_device;
    static std::unordered_map<string, shared_ptr<Op>> tensor_func_ops; //use for QNN

    Module() = default;
    virtual ~Module() = default;

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
        if (Module::backends.find(type) == Module::backends.end() || Module::backends[type] == nullptr) {
            switch (type) {
            case BackendType::MLLM_CPU: {
                shared_ptr<MemoryManager> mm = nullptr;
                mm = std::make_shared<SystemMemoryManager>();
                backends[MLLM_CPU] = new CPUBackend(mm);
                break;
            }
#ifdef USE_QNN
            case BackendType::MLLM_QNN: {
                //TODO: QNN
                shared_ptr<MemoryManager> mm = nullptr;
                mm = std::make_shared<SystemMemoryManager>();
                backends[MLLM_QNN] = new QNNBackend(mm);
            }
#endif
            default: {
            }
            }
        }
    }
    void to(BackendType backend_type) {
        initBackend(backend_type);
        backend_ = backends[backend_type];
        Module::tmp_device = backend_->type();
        Module::doToDevice = true;
        vector<Tensor> tmps;
        vector<int> tmpt;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor::gph_[std::to_string(i)] = Tensor(Module::backends[MLLM_CPU]);
            tmps.push_back(Tensor::gph_[std::to_string(i)]);
            tmpt.push_back(0);
        }
        vector<std::any> anyArgs = convertArgsToAnyVector(tmpt);
        Forward(tmps, anyArgs);
        Module::doToDevice = false;
        Module::tmp_device = MLLM_CPU;
        Tensor::gph_.clear();
    }

    BackendType device() const {
        return backend_->type();
    }
    
    static void initLoader(string path) {
        loader = new ParamLoader(std::move(path));
    }

    void load(string path) {
        initLoader(path);
        Module::doLoad = true;
        vector<Tensor> tmps;
        vector<int> tmpt;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor::gph_[std::to_string(i)] = Tensor(Module::backends[MLLM_CPU]);
            tmps.push_back(Tensor::gph_[std::to_string(i)]);
            tmpt.push_back(0);
        }
        vector<std::any> anyArgs = convertArgsToAnyVector(tmpt);
        Forward(tmps, anyArgs);
        Module::doLoad = false;
        Tensor::gph_.clear();
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs, vector<std::any> args) = 0;

    template <typename... Args>
    vector<std::any> convertArgsToAnyVector(Args... args) {
        return vector<std::any>{std::any(args)...};
    }
    template <typename... Args>
    vector<Tensor> operator()(vector<Tensor> inputs, Args... args) {
        vector<std::any> anyArgs = convertArgsToAnyVector(args...); 
        if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
#ifdef USE_QNN
            vector<shared_ptr<Tensor>> shared_inputs;
            vector<shared_ptr<Tensor>> shared_outputs;
            vector<shared_ptr<Tensor>> empty_v{};
            for (auto &input : inputs) {
                shared_inputs.push_back(std::shared_ptr<Tensor>(&input, [](Tensor *) {}));
            }
#endif

            for (auto &input : inputs) {
                input.setTtype(TensorType::NORMAL_TENSOR);
                input.status() = TENSOR_STATIC_INIT;
                if(input.batch() == 0){
                    Tensor::gph_[input.name()] = input;
                }
            }
            tensor_status = TENSOR_STATIC_INIT;
#ifdef USE_QNN
            Module::backends[inputs[0].device()]->onSetUpStart( shared_inputs, empty_v);
#endif
            auto result = Forward(inputs, anyArgs);
#ifdef USE_QNN          
            for (auto &output : result) {
                shared_outputs.push_back(std::shared_ptr<Tensor>(&output, [](Tensor *) {}));
            }
            Module::backends[result[0].device()]->onSetUpEnd( empty_v, shared_outputs);
#endif
            for (auto &input : inputs) {
                input.status() = TENSOR_STATIC_READY;
            }
            tensor_status = TENSOR_STATIC_READY;
#ifdef USE_QNN
            Module::backends[inputs[0].device()]->onExecuteStart( shared_inputs, empty_v);
#endif
            result = Forward(inputs, anyArgs);
#ifdef USE_QNN
            Module::backends[result[0].device()]->onExecuteEnd();
#endif
            return result;
        } else {
            return Forward(inputs, anyArgs);
        }
    }

    static int listIdx;
    static int runlistIdx;

    template <typename T>
    static vector<T > List(int n) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            modules.push_back(T());
            listIdx ++;
        }
        listIdx = 0;
        return modules;
    }

    // 递归终止函数
    template<typename T>
    static auto change_last(T value) {
        return std::make_tuple(value + std::to_string(listIdx) + ".");
    }
    // 递归函数
    template<typename T, typename... Args>
    static auto change_last(T head, Args... tail) {
        auto tail_tuple = change_last(tail...);
        return std::tuple_cat(std::make_tuple(head), tail_tuple);
    }
    template <typename T, typename... Args>
    static vector<T> List(int n, Args &&...args) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            auto new_args = change_last(args...);  // 创建新的参数包，最后一个参数被修改为原来的值+ std::to_string(listIdx)+ "."
            modules.push_back(std::move(T(std::apply([&](auto&&... args){ return T(std::forward<decltype(args)>(args)...); }, new_args))));
            listIdx++;
        }
        listIdx = 0;
        return modules;
    }
};

} // namespace mllm

#endif // MODULE_HPP
