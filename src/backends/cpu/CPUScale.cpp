
#include "CPUScale.hpp"

namespace mllm {

// template class CPUScale;
// template class CPUScale;


CPUScale::CPUScale(Backend *bn, string opName, float scale, float bias, bool bias_after_scale, bool multiThread)  :
    Op(bn, opName) {
    scale_ = scale;
    bias_ = bias;
    bias_after_scale_ = bias_after_scale;
    support_multi_thread_ = multiThread;
}

ErrorCode CPUScale::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUScale  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->shape(0), inputs[0]->shape(1), inputs[0]->shape(2), inputs[0]->shape(3));
    outputs[0]->setDtype(activationDtype());
    return NO_ERROR;
}

ErrorCode CPUScale::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    std::cout<<name() << "  CPUScale()" << std::endl;
    auto & input = inputs[0];
    auto & output = outputs[0];
    for(int n = 0; n<input->shape(0); ++n){
        for(int c = 0; c<input->shape(1); ++c){
            for(int h = 0; h<input->shape(2); ++h){
                for(int w = 0; w<input->shape(3); ++w){
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
    std::cout<<name() << "  CPUScale load" << std::endl;
    return NO_ERROR;
}
} // namespace mllm
