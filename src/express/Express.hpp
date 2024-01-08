#ifndef MLLM_EXPRESS_H
#define MLLM_EXPRESS_H

#include "ExpressBase.hpp"
#include "Types.hpp"
#include <string>
#include <vector>

using namespace mllm;
// 前置声明
//struct Context {
//    vector<NetParameter> sub_param_;
//    vector<NetOp *> net_ops;
//    std::set<NetTensor *> net_tensors;
//    int idx = 0;
//    int active_sub = 0;
//};
// NOLINTBEGIN(readability-identifier-naming)
void _SubgraphBegin(Context *ctx);
NetTensor *_Input(Context *ctx, vector<int> dims={}, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Parameter(Context *ctx, std::vector<NetTensor *> inputs, int batch, int seq, int head, int dim, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Add(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Causalmask(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_SiLU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Softmax(std::vector<NetTensor *> inputs, int axis, string name = "");
NetTensor *_Matmul(std::vector<NetTensor *> inputs,  bool transpose0, bool transpose1, string name = "");
NetTensor *_RMSNorm(std::vector<NetTensor *> inputs, int norm_size, float epsilon= 1e-6, string name = "");
NetTensor *_RoPE(std::vector<NetTensor *> inputs, int pose_type, string name = "");
NetTensor *_Scale(std::vector<NetTensor *> inputs, float scale, float bias, bool bias_after_scale, string name);
NetTensor *_Linear(std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
NetTensor *_Embedding(std::vector<NetTensor *> inputs, int vocab_size, int hidden_size, string name = "");
NetTensor *_Mul(std::vector<NetTensor *> inputs, string name = "");
// NetTensor *_View(std::vector<NetTensor *> inputs, vector<int> dims, vector<int>data_dims, string name = "");
NetTensor *_KVCache(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_ReLU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_ReLUSquaredActivation(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_GELU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_QuickGELU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_LayerNorm(std::vector<NetTensor *> inputs,int norm_size, bool bias= true,  float epsilon= 1e-6, string name = "");
vector<NetTensor *> _Split(std::vector<NetTensor *> inputs, int split_num, Chl split_dim, int split_dim_size = -1, string name = "");
NetTensor *_Gather(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Convolution2D(std::vector<NetTensor *> inputs,  int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias= false, string name = "");
NetTensor *_Convolution3D(std::vector<NetTensor *> inputs,  int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias= false, string name = "");
NetTensor *_AvgPool2D(std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
NetTensor *_MaxPool2D(std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
NetTensor *_Cat(std::vector<NetTensor *> inputs, Chl axis, string name = "");
// NetTensor *_Transpose(std::vector<NetTensor *> inputs, string name = "");
// NetTensor *_SubDim(std::vector<NetTensor *> inputs, Chl dim, vector<int> interval = {0, 0}, string name = "");
NetTensor *_Division(std::vector<NetTensor *> inputs, string name = "");
// NetTensor *_Norm(std::vector<NetTensor *> inputs, int L_n, string name = "");
// NOLINTEND(readability-identifier-naming)

#endif // MLLM_EXPRESS_H