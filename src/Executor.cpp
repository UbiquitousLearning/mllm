#include "Executor.hpp"
namespace mllm {
void Executor::init() {}
/*
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
 */
bool freeGraph = true;
void Executor::execute(shared_ptr<Tensor> input_tensor) {
    uint64_t t_start;
    uint64_t t_end;
    auto input_size = input_tensor->shape();
    bool init = false;
    bool reshape = false;
    checkReshape(init, reshape, input_size);
    //set Input tensor
    input_tensor->setName(net_->netParam()[0].net_tensors[0]->name);
    net_->tensors()[net_->netParam()[0].net_tensors[0]->name] = input_tensor;
    net_->subGraph()["G0"]->reflashInput(net_->tensors());
    for (int i = 0; i < (int)net_->subGraph().size(); ++i) {
        string name = "G" + std::to_string(i);
        auto &g = net_->subGraph()[name];
        if (init || reshape) {
            std::cout <<"["<< name << "]==== Reshape";
            t_start = mllm_time_us();
            g->reshape();
            g->setUpTensors();
            t_end = mllm_time_us();
            std::cout<<" ====  "<< (t_end - t_start)/1000.0F << " ms" << std::endl;
        }
        //load params
        if (init || freeGraph) {
            std::cout <<"["<< name << "]==== load" ;
            t_start = mllm_time_us();
            g->setUpOps(*data_loader_);
            t_end = mllm_time_us();
            std::cout<<"    ====  "<< (t_end - t_start)/1000.0F << " ms" << std::endl;
        }
        //exe
        std::cout <<"["<< name << "]==== execute" ;
        t_start = mllm_time_us();
        result_ = g->forward();
        t_end = mllm_time_us();
        std::cout<<" ====  "<< (t_end - t_start)/1000.0F << " ms" << std::endl;
        //free
        if(freeGraph) {
            std::cout <<"["<< name << "]==== free";
            t_start = mllm_time_us();
            g->freeOps();
            if (i < (int)net_->subGraph().size() - 1) {
                g->freeTensors();
            }
            net_->freeTensors(i);
            t_end = mllm_time_us();
            std::cout<<"    ====  "<< (t_end - t_start)/1000.0F << " ms" << std::endl;
        }
        std::cout <<"["<< name << "]==== end      === "<< result_[0]->name() << "'s shape:  [" << result_[0]->shape(0) << "," << result_[0]->shape(1) << "," << result_[0]->shape(2) << "," << result_[0]->shape(3) << "]" << std::endl;
    }
}

} // namespace mllm
