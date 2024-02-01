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

#include <memory/SystemMemoryManager.hpp>
#include <utility>

namespace mllm {

class Module {
public:
    static map<BackendType, Backend *> backends;
    static ParamLoader *loader;

    Module() = default;
    virtual ~Module() = default;

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
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
    static void initLoader(string path) {
        loader = new ParamLoader(std::move(path));
    }

    virtual vector<Tensor> Forward(vector<Tensor> inputs) = 0;

    vector<Tensor> operator()(vector<Tensor> inputs) {
        if (inputs[0].ttype() == TensorType::INPUT_TENSOR) {
            for (auto &input : inputs) {
                input.setTtype(TensorType::NORMAL_TENSOR);
                input.status() = TENSOR_STATIC_INIT;
            }

            Forward(inputs);
            for (auto &input : inputs) {
                input.status() = TENSOR_STATIC_SHAPED;
            }

            Forward(inputs);
            for (auto &input : inputs) {
                input.status() = TENSOR_STATIC_ALLOCED;
            }

            return Forward(inputs);
        } else {
            return Forward(inputs);
        }
    }

    vector<Tensor> call(vector<Tensor> inputs) {
        return operator()(inputs);
    }

    // template <typename T>
    // static vector<T *> List(int n) {
    //     static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
    //
    //     vector<T *> modules;
    //     for (int i = 0; i < n; i++) {
    //         modules.push_back(new T());
    //     }
    //     return modules;
    // }
    static int listIdx;

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

    template <typename T, typename... Args>
    static vector<T> List(int n, Args &&...args) {
        static_assert(std::is_base_of<Module, T>::value, "T must be a subclass of Module");
        listIdx = 0;
        vector<T> modules;
        for (int i = 0; i < n; i++) {
            modules.emplace_back(std::forward<Args>(args)...);
            listIdx++;
        }
        listIdx = 0;
        return modules;
    }
};

} // namespace mllm

#endif // MODULE_HPP
