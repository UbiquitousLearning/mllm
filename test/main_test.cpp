#include <iostream>
#include <valarray>
#include <csignal>
#include "cmdline.h"
#include "Net.hpp"
#include "Executor.hpp"
#include "express/Express.hpp"
#include "tokenizers/BPE/Bpe.hpp"
using namespace mllm;
#include "backends/cpu/compute/Matmul.hpp"
#define FP32(x) (MLLM_FP16_TO_FP32(x))
#define FP16(x) (MLLM_FP32_TO_FP16(x))

void setDatafp32(Tensor &t, const vector<float>& data){
    auto *dst = static_cast<float *>(t.rawHostPtr());
    for(auto d:data){
        *dst ++ = d;
    }
}

void setDatafp16(Tensor &t, const vector<float>& data){
    auto *dst = static_cast<mllm_fp16_t *>(t.rawHostPtr());
    for(auto d:data){
        *dst ++ = MLLM_FP32_TO_FP16(d);
    }
}

void test_fp32_mat_mul_sparse(){
    BackendConfig bn;
    Net net(bn);
    auto cpu_backend = net.backends()[BackendType::MLLM_CPU].get();

    Tensor x(cpu_backend);
    x.setName("x");
    x.reshape(2,1,2,2);
    x.setDtype(MLLM_TYPE_F32);
    x.alloc();
    setDatafp32(x, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0});
    x.printData<float>();

    Tensor W(cpu_backend);
    W.setName("W");
    W.reshape(1,1,2,2);
    W.setDtype(MLLM_TYPE_F32);
    W.alloc();
    setDatafp32(W, {1.0, 2.0, 3.0, 4.0});
    W.printData<float>();

    Tensor dst(x.shape());
    dst.setDtype(MLLM_TYPE_F32);
    dst.setBackend(cpu_backend);
    dst.setName("dst");
    dst.alloc();

    //    mat_mul_fp32(&x, &W, &dst, false, nullptr, false, true);
    mat_mul_sparse(&x, &W, &dst);
    dst.printData<float>();
}

void print_fp16_data(Tensor &t){
    printf("%s: ", t.name().c_str());
    printf("shape: [%d %d %d %d]\n",
           t.batch(), t.head(), t.sequence(), t.dimension());


    for(auto b=0;b < t.batch();b++){
        for(auto h=0;h < t.head();h++){

            for(auto i=0;i < t.sequence();i++){
                for(auto j = 0;j < t.dimension();j++){
                    printf("%f ", MLLM_FP16_TO_FP32(t.dataAt<mllm_fp16_t>(b,h,i,j)));
                }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }

}

void test_fp16_mat_mul_sparse(){
    BackendConfig bn;
    Net net(bn);
    auto cpu_backend = net.backends()[BackendType::MLLM_CPU].get();

    Tensor x(cpu_backend);
    x.setName("x");
    x.reshape(3,1,2,2);
    x.setDtype(MLLM_TYPE_F32);
    x.alloc();
    setDatafp32(x, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,0.0,0.0,10.0});
    x.printData<float>();

    Tensor W(cpu_backend);
    W.setName("W");
    W.reshape(1,1,2,2);
    W.setDtype(MLLM_TYPE_F16);
    W.alloc();
    setDatafp16(W, {1.0, 2.0, 3.0, 4.0});
    print_fp16_data(W);

    Tensor dst(x.batch(), x.head(),x.sequence(), W.dimension());
    dst.setDtype(MLLM_TYPE_F32);
    dst.setBackend(cpu_backend);
    dst.setName("dst");
    dst.alloc();

    //    mat_mul_fp32(&x, &W, &dst, false, nullptr, false, true);
    mat_mul_sparse(&x, &W, &dst);
    dst.printData<float>();
}

void test_fp16_mat_mul_sparse_avx2(){
    BackendConfig bn;
    Net net(bn);
    auto cpu_backend = net.backends()[BackendType::MLLM_CPU].get();

    Tensor x(cpu_backend);
    x.setName("x");
    x.reshape(1,1,1,10);
    x.setDtype(MLLM_TYPE_F32);
    x.alloc();
    setDatafp32(x, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    x.printData<float>();

    Tensor W(cpu_backend);
    W.setName("W");
    W.reshape(1,1,10,1);
    W.setDtype(MLLM_TYPE_F16);
    W.alloc();
    setDatafp16(W, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0});
    print_fp16_data(W);

    Tensor dst(x.batch(), x.head(),x.sequence(), W.dimension());
    dst.setDtype(MLLM_TYPE_F32);
    dst.setBackend(cpu_backend);
    dst.setName("dst");
    dst.alloc();

    //    mat_mul_fp32(&x, &W, &dst, false, nullptr, false, true);
    mat_mul_sparse(&x, &W, &dst);
    dst.printData<float>();
}

void test_mat_mul_id(){
    BackendConfig bn;
    Net net(bn);
    auto cpu_backend = net.backends()[BackendType::MLLM_CPU].get();

    Tensor x(cpu_backend);
    x.setName("x");
    x.reshape(1,1,2,3);
    x.setDtype(MLLM_TYPE_F32);
    x.alloc();
    setDatafp32(x, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    x.printData<float>();

    Tensor W(cpu_backend);
    W.setName("W");
    W.reshape(1,1,4,3);
    W.setDtype(MLLM_TYPE_F32);
    W.alloc();
    setDatafp32(W, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    W.printData<float>();

    Tensor dst(x.batch(), x.head(),x.sequence(), W.sequence());
    dst.setDtype(MLLM_TYPE_F32);
    dst.setBackend(cpu_backend);
    dst.setName("dst");
    dst.alloc();

    Tensor ids(dst.shape());
    ids.setDtype(MLLM_TYPE_F32);
    ids.setBackend(cpu_backend);
    ids.setName("ids");
    ids.alloc();
    setDatafp32(ids, {1.0,-1.0,1.0,1.0,
                      1.0,1.0,-1.0,1.0});
    ids.printData<float>();


    sparse_mat_mul_id(&x, &W, &ids, &dst);
    dst.printData<float>();
}

void test_mat_mul_id_fp16(){
    BackendConfig bn;
    Net net(bn);
    auto cpu_backend = net.backends()[BackendType::MLLM_CPU].get();

    Tensor x(cpu_backend);
    x.setName("x");
    x.reshape(1,1,2,3);
    x.setDtype(MLLM_TYPE_F32);
    x.alloc();
    setDatafp32(x, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    x.printData<float>();

    Tensor W(cpu_backend);
    W.setName("W");
    W.reshape(1,1,4,3);
    W.setDtype(MLLM_TYPE_F16);
    W.alloc();
    setDatafp16(W, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    print_fp16_data(W);

    Tensor dst(x.batch(), x.head(),x.sequence(), W.sequence());
    dst.setDtype(MLLM_TYPE_F32);
    dst.setBackend(cpu_backend);
    dst.setName("dst");
    dst.alloc();

    Tensor ids(dst.shape());
    ids.setDtype(MLLM_TYPE_F32);
    ids.setBackend(cpu_backend);
    ids.setName("ids");
    ids.alloc();
    setDatafp32(ids, {1.0,1.0,1.0,1.0,
                      1.0,1.0,1.0,1.0});
    ids.printData<float>();


    sparse_mat_mul_id(&x, &W, &ids, &dst);
    dst.printData<float>();
}

#include <chrono>
#include <random>
#include <type_traits>
#include <tuple>

using namespace std;

class Ticker{
public:
    Ticker() : t(std::chrono::high_resolution_clock::now()){}

    long Tick() {
        // 更新结束时间
        auto now = std::chrono::high_resolution_clock::now();

        // 计算开始和结束时间的差异，以纳秒为单位
        duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - t).count();

        // 更新开始时间为当前时间
        t = std::chrono::high_resolution_clock::now();

        return duration;
    }

    // 返回自上次Tick以来的微秒数
    double Microseconds() {
        return (double)duration / 1000.0;
    }

    // 返回自上次Tick以来的毫秒数
    double TickMilliseconds() {
        return (double)duration / 1000.0 / 1000.0;
    }


private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t;
    long duration; // nanoseconds
};

template<typename T>
void Randn(Tensor &t, float mean, float sigma){
    static std::mt19937 rng(std::random_device{}()); // 随机数生成器
    std::normal_distribution<double> dist(mean, sigma); // 正态分布

    for (auto b = 0; b < t.batch(); b++){
        for(auto h = 0; h < t.head(); h++) {
            for(auto s = 0; s < t.sequence(); s++){
                for(auto d = 0;d < t.dimension(); d++){
                    auto num = dist(rng);
                    if constexpr (std::is_same_v<T, float>) {
                        t.setDataAt<float>(b, h, s, d, (T)num);
                    }else if(std::is_same_v<T, mllm_fp16_t>) {
                        t.setDataAt<mllm_fp16_t>(b, h, s, d, MLLM_FP32_TO_FP16((float)num));
                    }else{
                        abort(); // not support yet
                    }
                }
            }
        }
    }
}

template<typename T>
void SetZero(Tensor &t, float rate) {
    static std::mt19937 rng(std::random_device{}()); // 随机数生成器
    std::uniform_real_distribution<double> dist(0.0, 1.0); // [0, 1)区间的均匀分布

    for (auto b = 0; b < t.batch(); ++b) {
        for (auto h = 0; h < t.head(); ++h) {
            for (auto s = 0; s < t.sequence(); ++s) {
                for (auto d = 0; d < t.dimension(); ++d) {
                    // 生成一个[0, 1)区间的随机数
                    auto rand_num = dist(rng);
                    // 如果随机数小于给定的比例rate，则将该位置的元素设置为0
                    if (rand_num < rate) {
                        // 根据Tensor中元素的数据类型来调用setDataAt方法
                        if constexpr (std::is_same_v<T, float>) {
                            t.setDataAt<float>(b, h, s, d, 0.0f);
                        } else if (std::is_same_v<T, mllm_fp16_t>) {
                            // 假设MLLM_FP32_TO_FP16(0.0f)能正确转换为对应的半精度浮点数0值
                            t.setDataAt<mllm_fp16_t>(b, h, s, d, MLLM_FP32_TO_FP16(0.0f));
                        } else {
                            abort(); // 目前不支持的数据类型
                        }
                    }
                }
            }
        }
    }
}

template<typename T1, typename T2>
bool Equal(Tensor &t1, Tensor &t2, Tensor *mask){
    auto B = t1.batch();
    auto H = t1.head();
    auto S = t1.sequence();
    auto D = t1.dimension();
    assert(B == t2.batch());
    assert(H == t2.head());
    assert(S == t2.sequence());
    assert(D == t2.dimension());
    if(mask != nullptr){
        assert(B == mask->batch());
        assert(H == mask->head());
        assert(S == mask->sequence());
        assert(D == mask->dimension());
    }

    for(auto b = 0;b < B; b++){
        for(auto h = 0; h < H; h++){
            for(auto s = 0;s < S;s++){
                for(auto d = 0;d < D;d++){
                    float tmp1, tmp2;
                    if constexpr (std::is_same_v<T1, float>){
                        tmp1 = t1.dataAt<float>(b, h, s, d);
                    }else if(std::is_same_v<T1, mllm_fp16_t >){
                        tmp1 = MLLM_FP16_TO_FP32(t1.dataAt<mllm_fp16_t>(b, h, s, d));
                    }else{
                        abort();
                    }

                    if constexpr (std::is_same_v<T2, float>){
                        tmp2 = t2.dataAt<float>(b, h, s, d);
                    }else if(std::is_same_v<T2, mllm_fp16_t >){
                        tmp2 = MLLM_FP16_TO_FP32(t2.dataAt<mllm_fp16_t>(b, h, s, d));
                    }else{
                        abort();
                    }

                    if(mask != nullptr) {
                        if (mask->dataAt<float>(b, h, s, d) > 0.0 && abs(tmp1 - tmp2) > 1e-3) {
                            fprintf(stderr, "not equal %lf != %lf\n", tmp1, tmp2);
                            return false;
                        }

                        if (mask->dataAt<float>(b, h, s, d) <= 0.0 && tmp1 != 0.0) {
                            fprintf(stderr, "%lf was masked\n", tmp1);
                            return false;
                        }
                    }else {
                        if(abs(tmp1 - tmp2) > 1e-3) {
                            fprintf(stderr, "not equal %lf != %lf\n", tmp1, tmp2);
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

template<typename T>
double SparsityRate(Tensor &t){
    // calculate the rate of elements that are smaller than 0.0 in the tensor
    auto B = t.batch();
    auto H = t.head();
    auto S = t.sequence();
    auto D = t.dimension();
    double rate = 0.0;
    for(auto b = 0;b < B; b++){
        for(auto h = 0; h < H; h++){
            for(auto s = 0;s < S;s++){
                for(auto d = 0;d < D;d++){
                    if constexpr (std::is_same_v<T, float>){
                        if(t.dataAt<float>(b, h, s, d) <= 0.0){
                            rate += 1.0;
                        }
                    }else if(std::is_same_v<T, mllm_fp16_t>){
                        if(MLLM_FP16_TO_FP32(t.dataAt<mllm_fp16_t>(b, h, s, d)) <= 0.0){
                            rate += 1.0;
                        }
                    }else{
                        abort();
                    }
                }
            }
        }
    }
    return rate / (B * H * S * D);
}

int thread_count = 8;

tuple<unique_ptr<Net>, unique_ptr<Executor>> LinearSparse(AbstructLoader &param_loader,int in_dim, int out_dim, string name){
    BackendConfig bn;
    auto net = make_unique<Net>(bn);
    std::unique_ptr<Context> ctx(new Context());
    auto c = ctx.get();
    auto *x = _Input(c);
    auto *out = _SparseLinear({x}, in_dim, out_dim, name);

    net->convert(ctx->sub_param_, BackendType::MLLM_CPU, thread_count);
    auto ex = make_unique<Executor>(&param_loader);
    ex->setup(net.get());  // load params
    return {std::move(net), std::move(ex)};
}

tuple<unique_ptr<Net>, unique_ptr<Executor>> SparseLinearId(AbstructLoader &param_loader,int in_dim, int out_dim, string name){
    BackendConfig bn;
    auto net = make_unique<Net>(bn);
    std::unique_ptr<Context> ctx(new Context());
    auto c = ctx.get();
    auto *x = _Input(c);
    auto *ids = _Input(c);
    auto *out = _SparseIdLinear({x, ids}, in_dim, out_dim, name);

    net->convert(ctx->sub_param_, BackendType::MLLM_CPU, thread_count);
    auto ex = make_unique<Executor>(&param_loader);
    ex->setup(net.get());  // load params
    return {std::move(net), std::move(ex)};
}

tuple<unique_ptr<Net>, unique_ptr<Executor>> Linear(AbstructLoader &param_loader,int in_dim, int out_dim, string name){
    BackendConfig bn;
    auto net = make_unique<Net>(bn);
    std::unique_ptr<Context> ctx(new Context());
    auto c = ctx.get();
    auto *x = _Input(c);
    auto *out = _Linear({x}, in_dim, out_dim, false, name);

    net->convert(ctx->sub_param_, BackendType::MLLM_CPU, thread_count);
    auto ex = make_unique<Executor>(&param_loader);
    ex->setup(net.get());  // load params
    return {std::move(net), std::move(ex)};
}

int N = 5;

bool test_sparse_id_linear(AbstructLoader &param_loader, int in_dim, int out_dim, const string &name){
    Ticker timer;

    auto [net_sparse, ex_sparse] = SparseLinearId(param_loader, in_dim, out_dim, name);
    auto [net, ex] = Linear(param_loader, in_dim, out_dim, name);

    auto cpu_backend = net_sparse->backends()[BackendType::MLLM_CPU].get();

    auto x = std::make_shared<Tensor>(cpu_backend);
    x->setName("x");
    x->reshape(1,1,N,in_dim);
    x->setDtype(MLLM_TYPE_F32);
    x->alloc();
    Randn<float>(*x, 0.0, 1.0);
    //    x->printData<float>();

    auto ids = std::make_shared<Tensor>(cpu_backend);
    ids->reshape(1,1,N,out_dim);
    ids->setDtype(MLLM_TYPE_F32);
    ids->setBackend(cpu_backend);
    ids->setName("ids");
    ids->alloc();
    Randn<float>(*ids, 0.0, 1.0);
    //    ids->printData<float>();

    timer.Tick();
    ex_sparse->execute(net_sparse.get(), {x, ids});
    timer.Tick();
    printf("\033[31m sparse linear id time: %.2lf us  (sparsity: %.2lf)\033[0m\n", timer.Microseconds(), SparsityRate<float>(*ids));
    auto res_sparse = ex_sparse->result()[0];
//        res_sparse->printData<float>();

    timer.Tick();
    ex->execute(net.get(), {x});
    timer.Tick();
    printf("\033[31m linear time: %.2lf us \033[0m\n", timer.Microseconds());
    auto res = ex->result()[0];
//        res->printData<float>();

    if(!Equal<float, float>(*res_sparse, *res, ids.get())) {
        printf("\033[31m not equal \033[0m\n");
        return false;
    } else {
        printf("\033[32m OK \033[0m\n");
        return true;
    }
}

bool test_sparse_linear(AbstructLoader &param_loader, int in_dim, int out_dim, const string &name){
    Ticker timer;

    auto [net_sparse, ex_sparse] = LinearSparse(param_loader, in_dim, out_dim, name);
    auto [net, ex] = Linear(param_loader, in_dim, out_dim, name);

    auto cpu_backend = net_sparse->backends()[BackendType::MLLM_CPU].get();

    auto x = std::make_shared<Tensor>(cpu_backend);
    x->setName("x");
    x->reshape(1,1,N,in_dim);
    x->setDtype(MLLM_TYPE_F32);
    x->alloc();
    Randn<float>(*x, 0.0, 1.0);
    SetZero<float>(*x, 0.5);
//    x->printData<float>();

    timer.Tick();
    ex_sparse->execute(net_sparse.get(), {x});
    timer.Tick();
    printf("\033[31m sparse linear time: %.2lf us  (sparsity: %.2lf)\033[0m\n", timer.Microseconds(), SparsityRate<float>(*x));
    auto res_sparse = ex_sparse->result()[0];
//    res_sparse->printData<float>();

    timer.Tick();
    ex->execute(net.get(), {x});
    timer.Tick();
    printf("\033[31m linear time: %.2lf us \033[0m\n", timer.Microseconds());
    auto res = ex->result()[0];
//    res->printData<float>();

    if(!Equal<float, float>(*res_sparse, *res, nullptr)) {
        printf("\033[31m not equal \033[0m\n");
        return false;
    } else {
        printf("\033[32m OK \033[0m\n");
        return true;
    }
}

int main(int argc, char **argv) {
    auto sparse_model_path = "./ReLULlama_new.mllm";
    auto model_path = "./ReLULlama.mllm";
    auto predictor_path = "./ReLULlama_predictor.mllm";

    Ticker timer;
    MultiFileParamLoader param_loader({model_path, sparse_model_path , predictor_path});
    MultiFileParamLoader q4_0_loader({"test_model_sparse_q4_0.mllm", "test_model_dense_q4_0.mllm"});
    MultiFileParamLoader q4_K_loader({"test_model_sparse_q4_K.mllm", "test_model_dense_q4_K.mllm"});
    timer.Tick();
    std::cout << timer.Microseconds() << " us" << std::endl;


    int hidden_size = 4096;
    int intermediate_size = 11008;

    assert(test_sparse_id_linear(q4_0_loader, hidden_size, intermediate_size, "up_proj"));
    assert(test_sparse_id_linear(q4_K_loader, hidden_size, intermediate_size, "up_proj"));
//    assert(test_sparse_linear(param_loader, intermediate_size, hidden_size, "down_proj"));
//    assert(test_sparse_id_linear(param_loader, hidden_size, intermediate_size, "up_proj"));
//    assert(test_sparse_linear(param_loader, intermediate_size, hidden_size, "model.layers.0.mlp.down_proj"));
//    assert(test_sparse_id_linear(param_loader, hidden_size, intermediate_size, "model.layers.0.mlp.up_proj"));

    return 0;
}