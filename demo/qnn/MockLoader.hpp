#ifndef MLLM_MockLoader_H
#define MLLM_MockLoader_H
#include "ParamLoader.hpp"
#include <cstdint>

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
        switch (tensor->dtype()) {
        case DataType::MLLM_TYPE_F32: {
            tensor->fullData<float>(2.f);
            break;
        }
        case DataType::MLLM_TYPE_I8: {
            tensor->fullData<int8_t>(2);
            break;
        }
        default:
            break;
        }
        return true;
    }
    bool load(std::shared_ptr<mllm::Tensor> tensor) override {
#ifdef DEBUG
        std::cout << "MockLoader load" << std::endl;
#endif
        return true;
    }
    DataType getDataType(string name) override {
        if (name.find("q8.weight") != string::npos) {
            std::cout << name << "int8" << std::endl;
            return DataType::MLLM_TYPE_I8;
        }
        return DataType::MLLM_TYPE_F32;
    }
};

} // namespace mllm

#endif // MLLM_MockLoader_H