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

#include <any>
#include <memory/SystemMemoryManager.hpp>
#include <utility>

namespace mllm {

class Module {
public:
    static map<BackendType, Backend *> backends;
    static ParamLoader *loader;
    static TensorStatus tensor_status;
    static bool doLoad;

    Module() = default;
    virtual ~Module() = default;

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
        if (Module::backends.find(type) == Module::backends.end()) {
            switch (type) {
            case BackendType::MLLM_CPU: {
                shared_ptr<MemoryManager> mm = nullptr;
                mm = std::make_shared<SystemMemoryManager>();
                backends[MLLM_CPU] = new CPUBackend(mm);
                break;
            }
            default: {
            }
            }
        }
    }
    void to(BackendType type) {
        initBackend(type);
    }
    static void initLoader(string path) {
        loader = new ParamLoader(std::move(path));
    }

    void load(string path) {
        initLoader(path);
        Module::doLoad = true;
        vector<Tensor> tmps;
        int max_in_size = 5;
        for (int i = 0; i < max_in_size; ++i) {
            Tensor::gph_[std::to_string(i)] = Tensor(Module::backends[MLLM_CPU]);
            tmps.push_back(Tensor::gph_[std::to_string(i)]);
        }
        vector<int> tmpt = {0, 0};
        operator()(tmps, tmpt);
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
        if(doLoad) {
            return Forward(inputs, anyArgs);
        }
        if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
            for (auto &input : inputs) {
                input.setTtype(TensorType::NORMAL_TENSOR);
                input.status() = TENSOR_STATIC_INIT;
                if(input.batch() == 0){
                    Tensor::gph_[input.name()] = input;
                }
            }
            tensor_status = TENSOR_STATIC_INIT;

            Forward(inputs, anyArgs);
            for (auto &input : inputs) {
                input.status() = TENSOR_STATIC_READY;
            }
            tensor_status = TENSOR_STATIC_READY;

            return Forward(inputs, anyArgs);
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
