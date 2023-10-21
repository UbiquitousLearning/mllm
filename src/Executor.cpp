#include "Executor.hpp"
namespace mllm {
void Executor::init() {
}

void Executor::execute(vector<int> input_size) {
    bool init = false;
    bool reshape = false;
    ;
    if (checkReshape(init, reshape, input_size)) {
        net_->reshapeInput(input_size);
    }
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << " Reshape" << std::endl;
        g->reshape(net_->tensors(), init, reshape, false);
    }
    net_->setInput();
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << "execute" << std::endl;
        result_ = g->forward(true);
        //result_[0]->printData<float>();
        std::cout << result_[0]->name() << "'s shape:  [" << result_[0]->shape(0) << "," << result_[0]->shape(1) << "," << result_[0]->shape(2) << "," << result_[0]->shape(3) << "]" << std::endl;
    }
}
void Executor::execute(shared_ptr<Tensor> input_tensor) {
    auto input_size = input_tensor->shape();
    bool init = false;
    bool reshape = false;
    checkReshape(init, reshape, input_size);
    input_tensor->setName(net_->netParam()[0].net_tensors[0]->name);
    net_->tensors()[net_->netParam()[0].net_tensors[0]->name] = input_tensor;
    //    net_->subGraph()["G0"]->reFlashInput(net_->tensors());
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        bool grapg0 = (i == 0);
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        std::cout << name << " Reshape" << std::endl;
        g->reshape(net_->tensors(), init, reshape, grapg0);
        if (data_loader_ != nullptr) {
            g->load(*data_loader_);
        }
        //exe
        std::cout << name << " execute" << std::endl;
        result_ = g->forward();
        //free
        std::cout << name << " free" << std::endl;
        g->free();
        net_->freeTensors(i);
        //result_[0]->printData<float>();
        std::cout << result_[0]->name() << "'s shape:  [" << result_[0]->shape(0) << "," << result_[0]->shape(1) << "," << result_[0]->shape(2) << "," << result_[0]->shape(3) << "]" << std::endl;
    }
}

} // namespace mllm
