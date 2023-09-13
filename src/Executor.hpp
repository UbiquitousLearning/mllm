#ifndef MLLM_EXECUTOR_H
#define MLLM_EXECUTOR_H
#include "Net.hpp"

namespace mllm {
    class Executor {
    public:
        Executor()=delete;
        Executor(Net *net):net_(net){
            // nothing to do
        }
        ~Executor()=default;
        

         /**
         * @brief 初始化
         * 使用几个线程，什么策略？
         */
        void Init();


        void GraphSetup(shared_ptr<Graph> subGraph)
        {
            // auto subGraph = net_.subGraphFP()[graph_name];
            subGraph->Setup();
        }

        /**
         * @brief 前行传播
         */
        const vector<shared_ptr<Tensor>>& GraphForward(shared_ptr<Graph> subGraph)
        {
            // auto subGraph = net_.subGraphFP()[graph_name];
            return subGraph->Forward();
            // TODO: 在此处插入 return 语句
        }

        void Execute();
    private:
        Net* net_;        
    };

}

#endif //MLLM_EXECUTOR_H
