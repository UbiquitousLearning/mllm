//
// Created by yirongjie.
//

#ifndef MLLM_GRAPH_H
#define MLLM_GRAPH_H
#include<map>
#include<unordered_map>
#include "NetParameter.hpp"
#include "Tensor.hpp"
#include "Op.hpp"
#include "backends/cpu/CPUMatmul.hpp" //TODO

using std::unordered_map;
// #include "layer.h"
namespace mllm {
    template <typename Dtype>
    class Graph {
    public:
        explicit Graph(const NetParameter& param, Backend* bn);
        virtual ~Graph() = default;

        /**
         * @brief 初始化
         */
        void Init(const NetParameter& in_param);//TODO?

        
        /**
         * @brief 初始化
         */
        void Setup();

        /**
         * @brief 前行传播
         */
        // const  vector<shared_ptr<Tensor<Dtype>>>& Forward();
        const  vector<shared_ptr<Tensor<Dtype>>>& Forward(Dtype* loss = NULL);
        // set input blobs then use Forward() instead.
        const  vector<shared_ptr<Tensor<Dtype>>>& Forward(const  vector<shared_ptr<Tensor<Dtype>>> & inTensors,
                                            Dtype* loss = NULL);

        /**
         * @brief 反向传播
         */
        void Backward();


    protected:
        Backend* backend_;
        // The network name
        string name_;
        // The phase: TRAIN or TEST
        // Phase phase_;

        // Individual layers in the net
        // vector<shared_ptr<Layer<Dtype> > > layers_;
        vector<string> layer_names_;

        // tensor indices for the input and the output of the net
        vector<int> input_tensor_indices_;
        vector<int> output_tensor_indices_;
        vector<Tensor<Dtype>*> input_tensors_;
        vector<Tensor<Dtype>*> output_tensors_;

        vector <string> op_names_;
        vector<vector<string>> op_in_names_;
	    unordered_map<string, vector<shared_ptr<Tensor<Dtype>>>> tensors_;   
        unordered_map<string, shared_ptr<Op<Dtype>>> ops_;
    };

}

#endif //MLLM_GRAPH_H
