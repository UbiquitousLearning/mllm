
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
    //std::cout<<name() << "  CPUScale  reshape" << std::endl;
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());
    //outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUScale::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUScale()" << std::endl;
    auto & input = inputs[0];
    auto & output = outputs[0];
    for(int n = 0; n<input->batch(); ++n){
        for(int c = 0; c<input->head(); ++c){
            for(int h = 0; h<input->sequence(); ++h){
                #pragma omp parallel for num_threads(4)
                for(int w = 0; w<input->dimension(); ++w){
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
    return Op::execute(inputs, outputs);
}

ErrorCode CPUScale::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    CHECK_EQ(inputs.size(), 1);
    CHECK_EQ(outputs.size(), 1);
    // outputs[0]->deepCopyFrom(inputs[0]);
    if(inputs[0]->masterTensor() == nullptr) {
        inputs[0]->free(); // TODO remove
    }
    outputs[0]->setDtype(activation_dtype());
    outputs[0]->alloc();
    inputs[0]->deepCopyFrom(outputs[0], false);
#ifdef DEBUG
    std::cout << "*"<<name()<<" setUp*" << std::endl;
#endif
    return NO_ERROR;
}

} // namespace mllm
