
#ifndef MLLM_CPUVIEW_H
#define MLLM_CPUVIEW_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUView final : public Op {
public:
    CPUView(Backend *bn, string opName, vector<int> dims, vector<int> data_dims, int threadCount);
    virtual ~CPUView() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int dim0_ = -1;
    int dim1_ = -1;
    int dim2_ = -1;
    int dim3_ = -1;
    // int dim4_ = -1; //only for BCTHW
    int data_dim0_;
    int data_dim1_;
    int data_dim2_;
    int data_dim3_;
    // int data_dim4_ = -999; //only for BCTHW
    int thread_count = 4;
    bool noNeedEx_ = false;
};

class CPUViewCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const {
        vector<int> dims = {(int)op_param["dim0"], (int)op_param["dim1"], (int)op_param["dim2"], (int)op_param["dim3"]};
        vector<int> data_dims = {(int)op_param["data_dim0"], (int)op_param["data_dim1"], (int)op_param["data_dim2"], (int)op_param["data_dim3"]};
        // if(op_param.find("dim4")!= op_param.end()) {
        //     dims.push_back((int)op_param["dim4"]);
        // }
        // if(op_param.find("data_dim4") != op_param.end()) {
        //     data_dims.push_back((int)op_param["data_dim4"]);
        // }
        assert(data_dims.size() == dims.size());
        return new CPUView(bn, name, dims, data_dims, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUVIEW_H
