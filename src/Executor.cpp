#include "Executor.hpp"
namespace mllm {
void Executor::init() {
}

void Executor::execute(vector<int> input_size) {
    bool init = false;
    bool reshape = false;
    if (input_size_.empty()) {
        input_size_ = input_size;
        init = true;
    } else if(input_size.empty()){
        reshape = false;
    } else if (input_size[0] == input_size_[0] && input_size[1] == input_size_[1] && input_size[2] == input_size_[2] && input_size[3] == input_size_[3]) {
        reshape = false;
    } else {
        input_size_ = input_size;
        reshape = true;
    }
    if (init || reshape) {
        net_->reshapeInput(input_size);
    }
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << " Reshape" << std::endl;
        if (init) {
            std::cout << "EXE:: Init" << std::endl;
            g->shapeInit(net_->tensors());
            g->setUp();
        } else if (reshape) {
            std::cout << "EXE:: Reshape" << std::endl;
            g->reshapeOutputs(net_->tensors());
        }
    }
    net_->setInput();
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << "execute" << std::endl;
        result_ = g->forward();
        //result_[0]->printData<float>();
        std::cout << result_[0]->name() << "'s shape:  [" << result_[0]->shape(0) << "," << result_[0]->shape(1) << "," << result_[0]->shape(2) << "," << result_[0]->shape(3) << "]" << std::endl;
    }
}
void Executor::execute(shared_ptr<Tensor> input_tensor) {
    auto input_size = input_tensor->shape();
    bool init = false;
    bool reshape = false;
    if (input_size_.empty()) {
        input_size_ = input_size;
        init = true;
    } else if(input_size.empty()){
        reshape = false;
    } else if (input_size[0] == input_size_[0] && input_size[1] == input_size_[1] && input_size[2] == input_size_[2] && input_size[3] == input_size_[3]) {
        reshape = false;
    } else {
        input_size_ = input_size;
        reshape = true;
    }
    input_tensor->setName(net_->netParam()[0].net_tensors[0]->name);
    net_->tensors()[net_->netParam()[0].net_tensors[0]->name] = input_tensor;
    net_->subGraph()["G0"]->reFlashInput(net_->tensors());
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << " Reshape" << std::endl;
        if (init) {
            std::cout << "EXE:: Init" << std::endl;
            g->shapeInit(net_->tensors());
            g->setUp();
        } else if (reshape) {
            std::cout << "EXE:: Reshape" << std::endl;
            g->reshapeOutputs(net_->tensors());
        }
    }
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << "execute" << std::endl;
        result_ = g->forward();
        //result_[0]->printData<float>();
        std::cout << result_[0]->name() << "'s shape:  [" << result_[0]->shape(0) << "," << result_[0]->shape(1) << "," << result_[0]->shape(2) << "," << result_[0]->shape(3) << "]" << std::endl;
    }
}

} // namespace mllm
