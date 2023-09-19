//
// Created by yirongjie.
//

#ifndef MLLM_GRAPH_H
#define MLLM_GRAPH_H
#include "NetParameter.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "ParamLoader.hpp"

// using std::unordered_map;
// #include "layer.h"
namespace mllm {
    
    class Graph {
    public:
        explicit Graph(const NetParameter& param, Backend* bn);
        virtual ~Graph() = default;

        /**
         * @brief 初始化
         */
        void Init();

        
        /**
         * @brief 初始化
         */
        void Setup();

        void Load(ParamLoader& loader);

        /**
         * @brief 前行传播
         */
        // const  vector<shared_ptr<Tensor>>& Forward();
        const  vector<shared_ptr<Tensor>>& Forward();
        // set input blobs then use Forward() instead.
        const  vector<shared_ptr<Tensor>>& Forward(const  vector<shared_ptr<Tensor>> & inTensors);

        /**
         * @brief 反向传播
         */
        void Backward();


    protected:
        NetParameter param_;
        Backend* backend_;
        // The network name
        string name_;
        // The phase: TRAIN or TEST
        // Phase phase_;

        // Individual layers in the net
        // vector<shared_ptr<Layer > > layers_;
        vector<string> layer_names_;

        // tensor indices for the input and the output of the net
        vector<int> input_tensor_indices_;
        vector<int> output_tensor_indices_;
        vector<Tensor*> input_tensors_;
        vector<Tensor*> output_tensors_;

        // vector <string> op_names_;
        // vector<vector<string>> op_in_names_;
	    unordered_map<string, vector<shared_ptr<Tensor>>> tensors_;  //opname: op's output Tensors 
        unordered_map<string, shared_ptr<Op>> ops_; //opname: op
    };

}

#endif //MLLM_GRAPH_H
