//
// Created by 30500 on 2020/12/2 0002.
//
#include "Graph.hpp"
#include "OP_defined.hpp"
namespace mllm {
    // template class Graph;
    // template class Graph;



    /**
     * @brief 初始化
     * @param in_param
     */
    
    Graph::Graph(const NetParameter &param,  Backend* bn)
    {
        backend_ = bn;
        param_ = param;
        Init();
    }
    
    void Graph::Init()
    {
        
        //RESHAPE
        tensors_["Input0"] = vector<shared_ptr<Tensor>>(1, NULL);
        for (auto& t: tensors_["Input0"]){
            std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(); 
            t = tensor1;
            t->SetByteWidth(sizeof(float));
            t->SetBackend(backend_);
            t->Reshape(1,3,5,5);//TODO Reshape  tensors_["input"] 
        }        
        for (int i = 0; i < (int)param_.net_ops.size(); ++i)
        {
            //TODO: 3改成不同的数
            auto net_op = param_.net_ops[i];
            auto op_name_ = net_op.name;
            tensors_[op_name_] = vector<shared_ptr<Tensor>>(3, NULL);
            for (auto& t: tensors_[op_name_]){
                std::shared_ptr<Tensor> tensor1 = std::make_shared<Tensor>(); 
                t = tensor1;
                t->SetByteWidth(sizeof(float));
                t->SetBackend(backend_);
            }
        }

        
        for (int i = 0; i < (int)param_.net_ops.size(); ++i)
        {
            auto net_op = param_.net_ops[i];
            shared_ptr<Op> myOp(NULL);
            auto newOp = backend_->OpCreate(net_op.param);
            myOp.reset(newOp);
            string lname = net_op.name;
            vector<string> inames = net_op.inOp;
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor>> inTensors;
            for (auto name: inames){
                inTensors.push_back(tensors_[name][0]);
            }
		    ops_[lname] = myOp;
            ops_[lname]->Reshape(inTensors, tensors_[lname]);//tensors_[lname]:1.Reshape
        }
    }

    
    void Graph::Setup()
    {
        for (auto& t: tensors_["Input0"]){
            t->Alloc();//to_cpu//malloc&memset 0 TODO
        }        
        for (int i = 0; i < (int)param_.net_ops.size(); ++i)
        {
            auto net_op = param_.net_ops[i];
            string lname = net_op.name;//op_names_[i];
            vector<string> inames = net_op.inOp;//op_in_names_[i];
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor>> inTensors;
            for (auto name: inames){
                inTensors.push_back(tensors_[name][0]);
            }
            ops_[lname]->Setup(inTensors, tensors_[lname]);//tensors_[lname]:malloc&memset 0 //TODO: 加入Bachend后改成不同Device的malloc
        }

    }
    
    /**
     * @brief 前向传播
     * @param loss
     * @return
     */
    
    const vector<shared_ptr<Tensor>> &Graph::Forward()
    {
        //TODO 改为递归

        for (int i = 0; i < (int)param_.net_ops.size(); ++i) 
        {
            auto net_op = param_.net_ops[i];
            string lname = net_op.name;//op_names_[i];
            vector<string> inames = net_op.inOp;//op_in_names_[i];
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor>> inTensors;
            for (auto name: inames){
                inTensors.push_back(tensors_[name][0]);
            }
            ops_[lname]->Execute(inTensors, tensors_[lname]);
        }
        //TODO
        return tensors_[param_.net_ops[param_.net_ops.size()-1].name];
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
     */
    
    void Graph::Backward() {

    }

}
