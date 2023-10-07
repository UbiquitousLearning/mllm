
#include "CPUScale.hpp"

namespace mllm {

// template class CPUScale;
// template class CPUScale;


CPUScale::CPUScale(Backend *bn, float scale, float bias, bool bias_after_scale, bool multiThread)  :
    Op(bn) {
    scale_ = scale;
    bias_ = bias;
    bias_after_scale_ = bias_after_scale;
    support_multi_thread_ = multiThread;
}

ErrorCode CPUScale::reshape(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUScale  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->num(), inputs[0]->channels(), inputs[0]->height(), inputs[0]->width());
    return NO_ERROR;
}

ErrorCode CPUScale::setUp(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUScale  setUp" << std::endl;
    if (!inputs[0]->allocted()) {
        inputs[0]->alloc(); // TODO remove
    }
    outputs[0]->alloc();
    return NO_ERROR;
}

ErrorCode CPUScale::execute(vector<shared_ptr<Tensor>> &inputs, vector<shared_ptr<Tensor>> &outputs) {
    std::cout << "CPUScale()" << std::endl;
    auto & input = inputs[0];
    auto & output = outputs[0];
    for(int n = 0; n<input->num(); ++n){
        for(int c = 0; c<input->channels(); ++c){
            for(int h = 0; h<input->height(); ++h){
                for(int w = 0; w<input->width(); ++w){
                    float value = input->dataAt<float>(n, c, h, w);
                    if(bias_after_scale_){
                        value = value * scale_ + bias_;
                    }else{
                        value = (value + bias_) * scale_;
                    }
                    output->setDataAt<float>(n, c, h, w, value);
                }
            }
        }
    }
    return NO_ERROR;
}

ErrorCode CPUScale::load(ParamLoader &loader) {
    std::cout << "CPUScale load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
