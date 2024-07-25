
#include "CPULinearSparse.hpp"
#include "Types.hpp"
#include "compute/VecDot.hpp"

namespace mllm {

ErrorCode sparse_mat_mul_fp32_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count, std::map<int, std::set<int>> validIndex, std::map<int, int> colToRow);

ErrorCode sparse_mat_mul_fp32_int8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count, std::map<int, std::set<int>> validIndex, std::map<int, int> colToRow);

CPULinearSparse::CPULinearSparse(Backend *bn, string opName, int in_features, int out_features, bool bias, int threadCount) :
    thread_count(threadCount),
    Op(bn, opName) {
    in_features_ = in_features;
    out_features_ = out_features;
    support_bias_ = bias;
    thread_count = threadCount;
    weight_.setBackend(bn);
    bias_.setBackend(bn);
}

ErrorCode CPULinearSparse::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // std::cout << name() << "  CPULinear  reshape" << std::endl;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    if (inputs[0]->count() == 0) {
        outputs[0]->reshape(0, 0, 0, 0);
        return Op::reshape(inputs, outputs);
    }
    // N     |    C       |   H                   |  W
    // -----------------------------------------------
    // 1     |out_channel | in_channel            |  1
    //       |out_features| in_features           |
    // -----------------------------------------------
    // batch |in_channel  | seq_len               |  1
    //       |in_features | inputs[0]->sequence()   |
    // -----------------------------------------------
    // batch |out_channel | seq_len               |  1
    //       |out_features|  inputs[0]->sequence()  |
    assert(inputs[0]->head() == 1);
    assert(in_features_ == inputs[0]->dimension());
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), out_features_);
    // outputs[0]->setDtype(activationDtype());
    return Op::reshape(inputs, outputs);
}

ErrorCode CPULinearSparse::load(AbstructLoader &loader) {
    this->loader_ = &loader;
    return Op::load(loader);
}

ErrorCode CPULinearSparse::dynamicLoad(AbstructLoader* loader, std::set<int> validCol) {
    // std::cout << name() << "  CPULinearSparse load" << std::endl;
    weight_.setName(name() + ".weight");
    weight_.reshape(1, 1, validCol.size(), out_features_);

    if (loader_->getDataType(weight_.name()) != MLLM_TYPE_COUNT) {
        weight_.setDtype(loader_->getDataType(weight_.name()));
        // 测试
        //  weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
        // validCol必须有序
        loader_->partialLoad(&weight_, validCol, in_features_, out_features_);
    } else {
        weight_.setDtype(MLLM_TYPE_F32);
        weight_.alloc();
    }
    if (support_bias_) {
        bias_.setName(name() + ".bias");
        bias_.reshape(1, 1, 1, out_features_);
        if (loader_->getDataType(bias_.name()) != MLLM_TYPE_COUNT) {
            bias_.setDtype(loader_->getDataType(bias_.name()));
            // bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
            loader_->load(&bias_);
        } else {
            bias_.setDtype(MLLM_TYPE_F32);
            bias_.alloc();
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode CPULinearSparse::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    if (inputs[0]->count() == 0) {
        return Op::execute(inputs, outputs);
    }

    // 提取稀疏化
    assert(inputs[0]->head() == 1);
    assert(in_features_ == inputs[0]->dimension());
    std::map<int, std::set<int>> validIndex;
    for (int i = 0; i < inputs[0]->sequence(); i++) {
        for (int j = 0; j < inputs[0]->dimension(); j++) {
            if (inputs[0]->dataAt<int>(0, 0, i, j) == 0) continue;
            validIndex[i].insert(j);
        }
    }
    // 有效的列，对应到模型是有效的行
    std::map<int, int> colToRow;
    std::set<int> validCol;
    for (const auto &index : validIndex) {
        for (auto col : index.second) validCol.insert(col);
    }
    for (const auto &element : validCol) colToRow[element] = colToRow.size();

    this->dynamicLoad(loader_, validCol);

    // 只实现MLLM_TYPE_F32
    switch (weight_.dtype()) {
    case MLLM_TYPE_F32: {
        sparse_mat_mul_fp32_fp32(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, thread_count, validIndex, colToRow);
        break;
    }
    case MLLM_TYPE_I8: {
        sparse_mat_mul_fp32_int8(inputs[0].get(), &weight_, outputs[0].get(), support_bias_, &bias_, thread_count, validIndex, colToRow);
        break;
    }
    default:
        break;
    }

    return Op::execute(inputs, outputs);
}

ErrorCode CPULinearSparse::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    weight_.free();
    if (support_bias_) {
        bias_.free();
    }
    return Op::free(inputs, outputs);
}

ErrorCode sparse_mat_mul_fp32_fp32(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count, std::map<int, std::set<int>> validIndex, std::map<int, int> colToRow) {
    const int M = src0->sequence();
    const int K = src0->dimension();
    const int N = src1->dimension();
    Tensor *src0_cal = src0;
    Tensor *src1_cal = src1;
    const int64_t blck_0 = 16;
    const int num_blocks = M / blck_0;
    const int remainder = M % blck_0;
#pragma omp parallel for num_threads(thread_count)
    for (int block = 0; block < num_blocks + 1; block++) {
        for (int m = block * blck_0; m < (block + 1) * blck_0 & m < num_blocks * blck_0 + remainder; m++) {
            for (int n = 0; n < N; n++) {
                *dst->ptrAt<float>(0, 0, m, n) = 0;
            }
            if (support_bias) {
                for (int n = 0; n < N; n++) {
                    *dst->ptrAt<float>(0, 0, m, n) += bias->dataAt<float>(0, 0, 0, n);
                }
            }
            if (validIndex.find(m) == validIndex.end()) continue;
            for (auto k : validIndex[m]) {
                vec_value_dot_fp32(N, dst->ptrAt<float>(0, 0, m, 0),
                                   src0_cal->dataAt<float>(0, 0, m, k),
                                   src1_cal->hostPtr<float>() + src1_cal->offset(0, 0, colToRow[k], 0), true);
            }
        }
    }
    return MLLM_NO_ERROR;
}

ErrorCode sparse_mat_mul_fp32_int8(Tensor *src0, Tensor *src1, Tensor *dst, bool support_bias, Tensor *bias, int thread_count, std::map<int, std::set<int>> validIndex, std::map<int, int> colToRow) {
    // TODO

    //     const int M = src0->sequence();
    //     const int K = src0->dimension();
    //     const int N = src1->dimension();
    //     Tensor *src0_cal = src0;
    //     Tensor *src1_cal = src1;
    //     const int64_t blck_0 = 16;
    //     const int num_blocks = M / blck_0;
    //     const int remainder = M % blck_0;
    // #pragma omp parallel for num_threads(thread_count)
    //     for (int block = 0; block < num_blocks + 1; block++) {
    //         for (int m = block * blck_0; m < (block + 1) * blck_0 & m < num_blocks * blck_0 + remainder; m++) {
    //             for (int n = 0; n < N; n++) {
    //                 *dst->ptrAt<float>(0, 0, m, n) = 0;
    //             }
    //             if (support_bias) {
    //                 for (int n = 0; n < N; n++) {
    //                     *dst->ptrAt<float>(0, 0, m, n) += bias->dataAt<float>(0, 0, 0, n);
    //                 }
    //             }
    //             if (validIndex.find(m) == validIndex.end()) continue;
    //             for (auto n : validIndex[m]) {
    //                 vec_value_dot_fp32(K, dst->ptrAt<float>(0, 0, m, 0),
    //                                         src0_cal->dataAt<float>(0, 0, m, n),
    //                                        src1_cal->hostPtr<float>() + src1_cal->offset(0, 0, colToRow[n], 0), true);
    //             }
    //         }
    //     }
    //     return MLLM_NO_ERROR;
}

} // namespace mllm
