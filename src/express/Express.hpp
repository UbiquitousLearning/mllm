#ifndef MLLM_EXPRESS_H
#define MLLM_EXPRESS_H

#include "NetParameter.hpp"
#include "Types.hpp"
#include <string>
#include <vector>

using namespace mllm;
// 前置声明
struct Context {
    vector<NetParameter> sub_param_;
    vector<NetOp *> net_ops;
    std::set<NetTensor *> net_tensors;
    int idx = 0;
    int active_sub = 0;
};
// NOLINTBEGIN(readability-identifier-naming)
void _SubgraphBegin(Context *ctx);
NetTensor *_Input(Context *ctx, vector<int> dims={}, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Parameter(Context *ctx, std::vector<NetTensor *> inputs, int batch, int seq, int head, int dim, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Add(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Causalmask(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_SiLU(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Softmax(Context *ctx, std::vector<NetTensor *> inputs, int axis, string name = "");
NetTensor *_Matmul(Context *ctx, std::vector<NetTensor *> inputs,  bool transpose0, bool transpose1, string name = "");
NetTensor *_RMSNorm(Context *ctx, std::vector<NetTensor *> inputs, int norm_size, string name = "");
NetTensor *_RoPE(Context *ctx, std::vector<NetTensor *> inputs, int pose_type, string name = "");
NetTensor *_Scale(Context *ctx, std::vector<NetTensor *> inputs, float scale, float bias, bool bias_after_scale, string name);
NetTensor *_Linear(Context *ctx, std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
NetTensor *_Embedding(Context *ctx, std::vector<NetTensor *> inputs, int vocab_size, int hidden_size, string name = "");
NetTensor *_Mul(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_View(Context *ctx, std::vector<NetTensor *> inputs, vector<int> dims, vector<int>data_dims, string name = "");
NetTensor *_KVCache(Context *ctx, std::vector<NetTensor *> inputs, bool isK, string name = "");
NetTensor *_ReLU(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_ReLUSquaredActivation(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_GELU(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_LayerNorm(Context *ctx, std::vector<NetTensor *> inputs,int norm_size, bool bias= true, string name = "");
vector<NetTensor *> _Split(Context *ctx, std::vector<NetTensor *> inputs, int split_num, Chl split_dim, int split_dim_size = -1, string name = "");
NetTensor *_Gather(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Convolution2D(Context *ctx, std::vector<NetTensor *> inputs,  int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias= false, string name = "");
NetTensor *_AvgPool2D(Context *ctx, std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
NetTensor *_MaxPool2D(Context *ctx, std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
NetTensor *_Cat(Context *ctx, std::vector<NetTensor *> inputs, Chl axis, string name = "");
NetTensor *_Transpose(Context *ctx, std::vector<NetTensor *> inputs, string name = "");
// NOLINTEND(readability-identifier-naming)

/*
// OLD VERSION
struct EOP {
    std::string name;
    std::vector<ETENSOR> connectedETensors;
    OpType type;
    OpParam op_param;

    EOP(std::string n, OpType t, OpParam op) :
        name(n), type(t), op_param(op) {
    }
};

// 定义 ETENSOR 结构体
struct ETENSOR {
    // std::string name;
    EOP op;

    ETENSOR(EOP o) :
        op(o) {
    }
    // ETENSOR(std::string n) : name(n), op(EOP("EOP0"))  {}
};
ETENSOR _EOP_(std::string name, OpType type, std::vector<ETENSOR> inputs, OpParam op_param);

ETENSOR _Input(vector<int> shape);
ETENSOR _Add(std::vector<ETENSOR> inputs);
ETENSOR _Mask(std::vector<ETENSOR> inputs);
ETENSOR _MatMul(std::vector<ETENSOR> inputs);
ETENSOR _RMSNorm(std::vector<ETENSOR> inputs);
ETENSOR _RoPE(std::vector<ETENSOR> inputs);
ETENSOR _Scale(std::vector<ETENSOR> inputs);
ETENSOR _SiLU(std::vector<ETENSOR> inputs);
ETENSOR _SoftMax(std::vector<ETENSOR> inputs, int axis);

void createNetParem(ETENSOR end_t, NetParameter &net_param);
*/
#endif // MLLM_EXPRESS_H