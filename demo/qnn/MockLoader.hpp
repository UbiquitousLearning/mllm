#ifndef MLLM_MockLoader_H
#define MLLM_MockLoader_H
#include "ParamLoader.hpp"

namespace mllm {
class MockLoader : public ParamLoader {
public:
    MockLoader(std::string filename) {
    }
    bool load(mllm::Tensor *tensor) override {
        std::cout << tensor->name() << std::endl;
#ifdef DEBUG
        std::cout << "MockLoader load" << std::endl;
#endif
        return true;
    }
    bool load(std::shared_ptr<mllm::Tensor> tensor) override {
#ifdef DEBUG
        std::cout << "MockLoader load" << std::endl;
#endif
        return true;
    }
    DataType getDataType(string name) override {
        if (name.find("wq.weight") != string::npos) {
            std::cout << name << "int8" << std::endl;
            return DataType::MLLM_TYPE_I8;
        }
        return DataType::MLLM_TYPE_F16;
    }
};

} // namespace mllm

#endif // MLLM_MockLoader_H