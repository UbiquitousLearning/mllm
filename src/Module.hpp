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
    static map<BackendType, Backend*> backends;
    static ParamLoader *loader;

    Module() = default;
    virtual ~Module() = default;

    static void initBackend(BackendType type = BackendType::MLLM_CPU) {
        switch (type){
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
        if(inputs[0].ttype() == TensorType::INPUT_TENSOR) {
            for (auto& input : inputs) {
                input.setTtype(TensorType::NORMAL_TENSOR);
                input.status() = TENSOR_STATIC_INIT;
            }

            Forward(inputs);
            for (auto& input : inputs) {
                input.status() = TENSOR_STATIC_ALLOCED;
            }

            return Forward(inputs);
        } else {
            return Forward(inputs);
        }
    }


};



} // namespace mllm


#endif //MODULE_HPP
