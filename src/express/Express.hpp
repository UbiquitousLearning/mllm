#ifndef MLLM_EXPRESS_H
#define MLLM_EXPRESS_H

#include "ExpressBase.hpp"
#include "Types.hpp"
#include <string>
#include <vector>
using namespace mllm;
void displayExpress(Context *c);

void _SubgraphBegin(Context *ctx, BackendType backend = MLLM_CPU);

NetTensor *_Input(Context *ctx, vector<int> dims = {}, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Parameter(Context *ctx, std::vector<NetTensor *> inputs, int batch, int seq, int head, int dim, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Range(Context *ctx, std::vector<NetTensor *> inputs, int start, int end, string name = "", DataType type = MLLM_TYPE_F32);
NetTensor *_Add(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Causalmask(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_SiLU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_SuperSiLU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Quantize(std::vector<NetTensor *> inputs, bool isNSHD = true, string name = "");
NetTensor *_Dequantize(std::vector<NetTensor *> inputs, bool isNSHD = true, string name = "", bool isFP32 = true);
NetTensor *_Softmax(std::vector<NetTensor *> inputs, int axis, int do_causal_mask, string name = "");
NetTensor *_Matmul(std::vector<NetTensor *> inputs, bool transpose0, bool transpose1, string name = "");
NetTensor *_RMSNorm(std::vector<NetTensor *> inputs, int norm_size, float epsilon = 1e-6, string name = "", bool isFP32 = true);
NetTensor *_RoPE(std::vector<NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384);
NetTensor *_IRoPE(std::vector<NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384);
NetTensor *_QNNRoPE(std::vector<NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384, bool isFP32 = true);
NetTensor *_QNNIRoPE(std::vector<NetTensor *> inputs, int pose_type, string name = "", int rope_theta = 10000, int max_position_embeddings = 16384, bool isFP32 = true);
NetTensor *_PositionalEmbedding(std::vector<NetTensor *> inputs, int max_num, int hidden_dim, string name = "");
NetTensor *_Scale(std::vector<NetTensor *> inputs, float scale, float bias, bool bias_after_scale, string name);
NetTensor *_Linear(std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
NetTensor *_LinearINT8(std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
vector<NetTensor *> _LinearINT8ShadowMerge(std::vector<NetTensor *> inputs, int in_features, int out_features, bool bias, string name = "");
NetTensor *_LinearINT8ShadowCPU(std::vector<NetTensor *> inputs, int in_features, int out_features, int max_position = 1024, bool bias = false, string name = "");
NetTensor *_Embedding(std::vector<NetTensor *> inputs, int vocab_size, int hidden_size, string name = "");
NetTensor *_Mul(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_KVCache(std::vector<NetTensor *> inputs, int cache_max, string name = "");
NetTensor *_KVCache(std::vector<NetTensor *> inputs, int n_rep, int cache_max, string name = "");
NetTensor *_KVCacheNPU(std::vector<NetTensor *> inputs, int cache_max, string name = "");
NetTensor *_KVCache(std::vector<NetTensor *> inputs, int n_rep, bool share_input, int cache_max, string name = "");
NetTensor *_ReLU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_ReLUSquaredActivation(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_GELU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_QuickGELU(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_LayerNorm(std::vector<NetTensor *> inputs, int norm_size, bool bias = true, float epsilon = 1e-6, string name = "");
vector<NetTensor *> _Split(std::vector<NetTensor *> inputs, int split_num, Chl split_dim, int split_dim_size = -1, string name = "");
NetTensor *_Gather(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Convolution2D(std::vector<NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias = false, string name = "");
NetTensor *_Convolution3D(std::vector<NetTensor *> inputs, int in_channel, int out_channel, vector<int> kernal, vector<int> stride, PaddingType padding, bool bias = false, string name = "");
NetTensor *_AvgPool2D(std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
NetTensor *_MaxPool2D(std::vector<NetTensor *> inputs, vector<int> kernal, vector<int> stride, PaddingType padding, string name = "");
NetTensor *_Cat(std::vector<NetTensor *> inputs, Chl axis, string name = "");
NetTensor *_Division(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_Replace(std::vector<NetTensor *> inputs, string name = "");
NetTensor *_SparseLinear(std::vector<NetTensor *> inputs, int in_dim, int out_dim, string name = "");
NetTensor *_SparseIdLinear(std::vector<NetTensor *> inputs, int in_dim, int out_dim, string name = "");
NetTensor *_Predictor(std::vector<NetTensor *> inputs, int in_dim, int out_dim, string name = "");
NetTensor *_WNop(std::vector<NetTensor *> inputs, int sync_type, string name = "");
vector<NetTensor *> _MergeOutput(std::vector<NetTensor *> inputs, string name = "");
vector<NetTensor *> _SplitInput(std::vector<NetTensor *> inputs, bool isPrompt, int num, string name = "");
NetTensor *_Transpose(std::vector<NetTensor *> inputs, std::vector<int> perm, string name = "");

#endif // MLLM_EXPRESS_H