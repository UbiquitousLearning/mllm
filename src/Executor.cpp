#include "Executor.hpp"
namespace mllm {
void Executor::init() {
}

void Executor::execute(vector<int> input_size ) {
    bool reshape = false;
    if(input_size_.empty()){
        input_size_ = input_size;
        reshape = true;
    } else if(input_size[0] == input_size_[0] && input_size[1] == input_size_[1] && input_size[2] == input_size_[2] && input_size[3] == input_size_[3]){
        reshape = false;
    } else {
        input_size_ = input_size;
        reshape = true;
    }
    if(reshape){
        net_->reshapeInput(input_size[0], input_size[1], input_size[2], input_size[3]);
    }
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << std::endl;
        if(reshape) {
            std::cout<<"EXE:: reshape"<<std::endl;
            graphReshape(g, net_->tensors());
            graphSetUp(g);
        }
        auto result = graphForward(g);
    }
}

} // namespace mllm
