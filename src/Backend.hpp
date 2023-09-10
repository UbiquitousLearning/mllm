#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "Graph.hpp"
namespace mllm
{
    class Backend {
    public:
        Backend()= default;
        virtual ~Backend() = default;

        bool testOperator(shared_ptr<Op<float>> op) {
            // return OPMap.contains(op->type);
        }
        
        bool testOperator(shared_ptr<Op<int8_t>> op) {
            // return OPMap.contains(op->type);
        }

        void Execute() {

        }
    private:
        //

    };
    
} // namespace mllm



#endif //MLLM_BACKEND_H