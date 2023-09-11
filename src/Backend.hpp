#ifndef MLLM_BACKEND_H
#define MLLM_BACKEND_H

#include "Graph.hpp"
namespace mllm
{
    class Backend {
    public:
        Backend()= default;
        virtual ~Backend() = default;

        bool CheckSupport(shared_ptr<Op<float>> op) {
            // return OPMap.contains(op->type);
            return true;
        }
        
        bool CheckSupport(shared_ptr<Op<int8_t>> op) {
            // return OPMap.contains(op->type);
            return true;
        }

        void Execute() {

        }

        bool CPUTensorConvert(shared_ptr<Tensor<float>> src_tensor, shared_ptr<Tensor<float>> dst_tensor, int type_); //NCHW --> NHWC ..., TODO type_:enum
    private:
        //

    };
    
} // namespace mllm



#endif //MLLM_BACKEND_H