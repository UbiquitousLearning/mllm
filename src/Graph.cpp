//
// Created by 30500 on 2020/12/2 0002.
//
#include "Graph.hpp"
namespace mllm {
    template class Graph<float>;
    template class Graph<int8_t>;



    /**
     * @brief 初始化
     * @tparam Dtype
     * @param in_param
     */
    template <typename Dtype>
    Graph<Dtype>::Graph(const NetParameter &param)
    {
        Init(param);
    }
    template <typename Dtype>
    void Graph<Dtype>::Init(const NetParameter &in_param)
    {
        //init from files
        //init from code
        op_names_ = in_param.op_names_;
        op_in_names_ = in_param.op_in_names_;
    }

    template <typename Dtype>
    void Graph<Dtype>::Setup()
    {
        auto bn = std::shared_ptr<Backend>(new Backend());
        tensors_["input"] = vector<shared_ptr<Tensor<Dtype>>>(1, NULL);
        for (auto& t: tensors_["input"]){
            std::shared_ptr<Tensor<Dtype>> tensor1 = std::make_shared<Tensor<Dtype>>(); 
            t = tensor1;
            t->SetBackend(bn);
            t->Reshape(1,3,5,5);//TODO Reshape  tensors_["input"] 
            t->Alloc();//to_cpu//malloc&memset 0 TODO
        }        
        for (int i = 0; i < (int)op_names_.size(); ++i)
        {
            //TODO: 3改成不同的数
            tensors_[op_names_[i]] = vector<shared_ptr<Tensor<Dtype>>>(3, NULL);
            for (auto& t: tensors_[op_names_[i]]){
                std::shared_ptr<Tensor<Dtype>> tensor1 = std::make_shared<Tensor<Dtype>>(); 
                t = tensor1;
            }
        }
        for (int i = 0; i < (int)op_names_.size(); ++i)
        {
            shared_ptr<Op<Dtype>> myOp(NULL);
            myOp.reset(new CPUMatmul<Dtype>(mllm_CPU,true,true,true,true));	//TODO
            string lname = op_names_[i];
            vector<string> inames = op_in_names_[i];
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor<Dtype>>> inTensors;
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
    template <typename Dtype>
    const vector<shared_ptr<Tensor<Dtype>>> &Graph<Dtype>::Forward(Dtype *loss)
    {
        //TODO 改为递归

        for (int i = 0; i < (int)op_names_.size(); ++i) 
        {
            string lname = op_names_[i];
            vector<string> inames = op_in_names_[i];
            //TODO: CHECK一下 inTensors 尤其是[0]
            vector<shared_ptr<Tensor<Dtype>>> inTensors;
            for (auto name: inames){
                inTensors.push_back(tensors_[name][0]);
            }
            ops_[lname]->Execute(inTensors, tensors_[lname]);
        }
        //TODO
        return tensors_[op_names_[op_names_.size()-1]];
    }

    template<typename Dtype>
    const vector<shared_ptr<Tensor<Dtype>>> &Graph<Dtype>::Forward(const  vector<shared_ptr<Tensor<Dtype>>> &inTensors, Dtype *loss) {
        // Copy 
        for (int i = 0; i < inTensors.size(); ++i) {
            input_tensors_[i]->CopyFrom(*inTensors[i]);
        }
        return Forward(loss);
    }

    /**
     * @brief 反向传播
     * @tparam Dtype
     */
    template<typename Dtype>
    void Graph<Dtype>::Backward() {

    }

}
