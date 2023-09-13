//
// Created by 30500 on 2020/12/2 0002.
//
#include "Graph.hpp"
namespace mllm {
    // template class Graph;
    // template class Graph;



    /**
     * @brief 初始化
     * @tparam Dtype
     * @param in_param
     */
    
    Graph::Graph(const NetParameter &param,  Backend* bn)
    {
        backend_ = bn;
        Init(param);
    }
    
    void Graph::Init(const NetParameter &in_param)
    {
        //init from files
        //init from code
        op_names_ = in_param.op_names_;
        op_in_names_ = in_param.op_in_names_;
    }

    
    void Graph::Setup()
    {
        // auto bn = new Backend();
        tensors_["input"] = vector<shared_ptr<Tensor>>(1, NULL);
        for (auto& t: tensors_["input"]){
            std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(); 
            t = tensor1;
            t->SetBackend(backend_);
            t->Reshape(1,3,5,5);//TODO Reshape  tensors_["input"] 
            t->Alloc();//to_cpu//malloc&memset 0 TODO
        }        
        for (int i = 0; i < (int)op_names_.size(); ++i)
        {
            //TODO: 3改成不同的数
            tensors_[op_names_[i]] = vector<shared_ptr<Tensor>>(3, NULL);
            for (auto& t: tensors_[op_names_[i]]){
                std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(); 
                t = tensor1;
                t->SetBackend(backend_);
            }
        }
        for (int i = 0; i < (int)op_names_.size(); ++i)
        {
            shared_ptr<Op> myOp(NULL);
            myOp.reset(new CPUMatmul(backend_,true,true,true,true));	//TODO
            string lname = op_names_[i];
            vector<string> inames = op_in_names_[i];
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor>> inTensors;
            for (auto name: inames){
                inTensors.push_back(tensors_[name][0]);
            }
		    ops_[lname] = myOp;
            ops_[lname]->Setup(inTensors, tensors_[lname]);//tensors_[lname]:1.Reshape 2.malloc&memset 0 //TODO: 加入Bachend后改成不同Device的malloc
        }

    }
    
    /**
     * @brief 前向传播
     * @tparam Dtype
     * @param loss
     * @return
     */
    
    const vector<shared_ptr<Tensor>> &Graph::Forward()
    {
        //TODO 改为递归

        for (int i = 0; i < (int)op_names_.size(); ++i) 
        {
            string lname = op_names_[i];
            vector<string> inames = op_in_names_[i];
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor>> inTensors;
            for (auto name: inames){
                inTensors.push_back(tensors_[name][0]);
            }
            ops_[lname]->Execute(inTensors, tensors_[lname]);
        }
        //TODO
        return tensors_[op_names_[op_names_.size()-1]];
    }

    
    const vector<shared_ptr<Tensor>> &Graph::Forward(const vector<shared_ptr<Tensor>> &inTensors) {
        // Copy 
        for (int i = 0; i < inTensors.size(); ++i) {
            input_tensors_[i]->CopyFrom(*inTensors[i]);
        }
        return Forward();
    }

    /**
     * @brief 反向传播
     * @tparam Dtype
     */
    
    void Graph::Backward() {

    }

}
