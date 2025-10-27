
#ifndef MLLM_QNNSPLIT_H
#define MLLM_QNNSPLIT_H

#include "QNNCommonOp.hpp"
#include "Types.hpp"
namespace mllm {
class QNNSplit final : public QNNCommonOp {
public:
    QNNSplit(Backend *bn, string opName, int splitNum, Chl splitDim, int splitDimSize, std::vector<int> each_dims = {});
    virtual ~QNNSplit() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int split_num_;
    Chl split_dim_;
    int split_dim_size_;
    std::vector<int> each_dims_;
};

class QNNSplitCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        int splitNum = (int)op_param["split_num"];
        Chl splitDim = (Chl)op_param["split_dim"];
        int splitDimSize;
        if (op_param.find("split_dim_size") != op_param.end()) {
            splitDimSize = (int)op_param["split_dim_size"];
        } else {
            splitDimSize = -1;
        }

        // if using each_dim
        std::vector<int> each_dims = {};
        if (splitDimSize == -1) {
            int cnt = 0;
            while (true) {
                auto iter = op_param.find("split_dim_size_" + std::to_string(cnt++));
                if (iter == op_param.end()) break;
                each_dims.push_back((int)iter->second);
            }
        }
        return new QNNSplit(bn, name, splitNum, splitDim, splitDimSize, each_dims);
    }
};

} // namespace mllm

#endif
