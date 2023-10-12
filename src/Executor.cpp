#include "Executor.hpp"
namespace mllm {
void Executor::init() {
}

void Executor::execute(vector<int> input_size ) {
    bool init = false;
    bool reshape = false;
    if(input_size_.empty()){
        input_size_ = input_size;
        init = true;
    } else if(input_size[0] == input_size_[0] && input_size[1] == input_size_[1] && input_size[2] == input_size_[2] && input_size[3] == input_size_[3]){
        reshape = false;
    } else {
        input_size_ = input_size;
        reshape = true;
    }
    if(init || reshape){
        net_->reshapeInput(input_size[0], input_size[1], input_size[2], input_size[3]);
    }
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << std::endl;
        if(init) {
            std::cout<<"EXE:: Init"<<std::endl;
            graphShapeInit(g, net_->tensors());
            graphSetUp(g);
        } else if (reshape) {
            std::cout<<"EXE:: Reshape"<<std::endl;
            graphReshapeOutputs(g, net_->tensors());
        }
        auto result = graphForward(g);
//        result[0]->printData<float>();
        std::cout<<result[0]->name()<<"'s shape:  ["<<result[0]->shape(0)<<","<<result[0]->shape(1)<<","<<result[0]->shape(2)<<","<<result[0]->shape(3)<<"]"<<std::endl;
    }
}

} // namespace mllm
