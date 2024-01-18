#include <iostream>
#include <valarray>
#include <csignal>
#include "Net.hpp"
#include "Executor.hpp"
#include "NetParameter.hpp"
#include "QnnTypes.h"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
#include "backends/QNN/QNNBackend.hpp"
#include "memory/SystemMemoryManager.hpp"
#include "backends/QNN/op/QNNAdd.hpp"
#include "MockLoader.hpp"

#include <iostream>
#include <fstream>

using namespace mllm;

NetTensor * RoPE(Context *ctx,  uint32_t hidden_dim, uint32_t ffn_hidden_dim) {

    auto *i = _Input(ctx);
    auto *z = _RoPE(ctx, {i}, "ffn.rope");

    return z;
}


void textTensor(shared_ptr<Tensor> input_tensor, Net &net, vector<int> shape) {

    std::ifstream file("QNNRoPETest.input.output.txt"); // 打开文件
    if (!file.is_open()) { // 检查文件是否成功打开
        std::cout << "无法打开文件" << std::endl;
    }

    input_tensor->setBackend(net.backends()[BackendType::MLLM_CPU].get());
    input_tensor->reshape(shape);
    input_tensor->alloc();
    input_tensor->fullData<float>(1);

    int batch = shape[0];
    int n1 = shape[1];
    int n2 = shape[2];
    int n3 = shape[3];
    for (int n = 0; n < batch; n++) {
        for (int c = 0; c < n1; c++) {
            for (int h = 0; h < n2; h++) {
                for (int w = 0; w < n3; w++) {
                    float ff;
                    if (!(file >> ff )) { // 逐个读取 float 数据到数组中
                        std::cout << "读取文件失败" << std::endl;
                    }
                    input_tensor->setDataAt<float>(n, c, h, w, ff);
                }
            }
        }
    }
}


int main() {
    int vocab_size = 32000;
    int hidden_dim = 4096;
    int ffn_hidden_dim = 11008;
    int mutil_head_size = 32;

    std::unique_ptr<Context> c_ptr(new Context());
    auto *c = c_ptr.get();

    RoPE(c, hidden_dim, ffn_hidden_dim);

    BackendConfig bn;
    Net net(c->sub_param_, bn);
    net.convert(c->sub_param_, MLLM_QNN);
    std::cout << "convert done" << std::endl;

    MockLoader loader("");
    Executor ex(&loader);
    shared_ptr<Tensor> input = std::make_shared<Tensor>();

    // 1 batch seqence length embedding
    textTensor(input, net, {1, 180, 1, 128});

    ex.execute(&net, input);
    // ex.perf();
    auto result = ex.result();
    result[0]->printData<float>();
}