#ifndef MLLM_QNNOPTNET_H
#define MLLM_QNNOPTNET_H

#include "Op.hpp"
#include "ParamLoader.hpp"
#include "Graph.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "Net.hpp"
#include "express/ExpressBase.hpp"

namespace mllm {
class QNNOptNet : public Net {
public:
    explicit QNNOptNet(BackendConfig config, Context *ctx);
    virtual ~QNNOptNet() = default;

    virtual void convert(vector<NetParameter> &param, BackendType backend_type = BackendType::MLLM_QNN, int threadCount = 4) override;
    virtual void convert(Context* ctx, BackendType backend_type = BackendType::MLLM_QNN, int threadCount = 4);

private:
    void build_new_graph(std::vector<NetTensor *> inputs, NetOp *op);
    Context *ctx_;
    std::string Quantizationtype = "Smoothquant";
};

} // namespace mllm

#endif // MLLM_NET_H