
#include "QNNSplit.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cassert>
#include <cstdint>

namespace mllm {
QNNSplit::QNNSplit(Backend *bn, string opName, int splitNum, Chl splitDim, int splitDimSize, std::vector<int> each_dims) :
    split_num_(splitNum), split_dim_(splitDim), split_dim_size_(splitDimSize), each_dims_(each_dims), QNNCommonOp(bn, opName) {
}

ErrorCode QNNSplit::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(split_num_ == outputs.size());
    assert(inputs.size() == 1);
    switch (split_dim_) {
    case Chl::HEAD: {
        switch (split_dim_size_) {
        case -1: /*using each_dims*/ {
            // check shape
            assert(!each_dims_.empty() && "split op with split_dims_size_ == 1 should has each_dims_ params");
            {
                int head_sum = 0;
                for (auto item : each_dims_) head_sum += item;
                assert(head_sum == inputs[0]->head() && "sum(each_dims_) miss match inputs[0]'s head dim");
            }
            assert(outputs.size() == each_dims_.size() && "outputs size miss match each_dims_ size");

            // reshape output
            for (size_t i = 0; i < each_dims_.size(); ++i) {
                outputs[i]->reshape(inputs[0]->batch(), each_dims_[i], inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        default: /*split for same size*/ {
            assert(inputs[0]->head() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head() / split_num_, inputs[0]->sequence(), inputs[0]->dimension());
            }
            break;
        }
        }
        break;
    }
    case Chl::SEQUENCE: {
        switch (split_dim_size_) {
        case -1: /*using each_dims*/ {
            // check shape
            assert(!each_dims_.empty() && "split op with split_dims_size_ == 1 should has each_dims_ params");
            {
                int seq_sum = 0;
                for (auto item : each_dims_) seq_sum += item;
                assert(seq_sum == inputs[0]->sequence() && "sum(each_dims_) miss match inputs[0]'s sequence dim");
            }
            assert(outputs.size() == each_dims_.size() && "outputs size miss match each_dims_ size");

            // reshape output
            for (size_t i = 0; i < each_dims_.size(); ++i) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), each_dims_[i], inputs[0]->dimension());
            }
            break;
        }
        default: {
            assert(inputs[0]->sequence() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / split_num_, inputs[0]->dimension());
            }
            break;
        }
        }
        break;
    }
    case Chl::DIMENSION: {
        switch (split_dim_size_) {
        case -1: /*using each_dims*/ {
            // check shape
            assert(!each_dims_.empty() && "split op with split_dims_size_ == 1 should has each_dims_ params");
            {
                int dimension_sum = 0;
                for (auto item : each_dims_) dimension_sum += item;
                assert(dimension_sum == inputs[0]->sequence() && "sum(each_dims_) miss match inputs[0]'s dimension dim");
            }
            assert(outputs.size() == each_dims_.size() && "outputs size miss match each_dims_ size");

            // reshape output
            for (size_t i = 0; i < each_dims_.size(); ++i) {
                outputs[i]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), each_dims_[i]);
            }
            break;
        }
        default: {
            assert(inputs[0]->dimension() % split_num_ == 0);
            for (auto &output : outputs) {
                output->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension() / split_num_);
            }
            break;
        }
        }
        break;
    }
    case Chl::D_HD: {
        assert(inputs[0]->dimension() % split_num_ == 0);
        for (auto &output : outputs) {
            output->reshape(inputs[0]->batch(), split_dim_size_, inputs[0]->sequence(), inputs[0]->dimension() / (split_num_ * split_dim_size_));
        }
        break;
    }
    case Chl::HD: {
        assert(inputs[0]->dimension() % split_num_ == 0);
        for (auto &output : outputs) {
            output->reshape(inputs[0]->batch(), split_dim_size_, inputs[0]->sequence(), inputs[0]->dimension() / (split_num_ * split_dim_size_));
        }
        break;
    }
    default: {
        break;
    }
    }

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNSplit::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    for(auto &output : outputs) {
        output->setDtype(inputs[0]->dtype());
    }
    vector<uint32_t> split_index(split_num_ - 1);
    for (int i = 0; i < split_num_; i++) {
        split_index[i] = split_dim_size_ * (i + 1);
    }

    uint32_t split_index_dim[1] = {2};
    auto paramTensorName = name() + ".param";
    vector<Qnn_Param_t> params = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "axis",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = 3}}},
        {.paramType = QNN_PARAMTYPE_TENSOR,
         .name = "split_index",
         .tensorParam = (Qnn_Tensor_t){.version = QNN_TENSOR_VERSION_1,
                                       .v1 = {
                                           .id = 0,
                                           .name = paramTensorName.c_str(),
                                           .type = QNN_TENSOR_TYPE_STATIC,
                                           .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                           .dataType = QNN_DATATYPE_UINT_32,
                                           .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                              QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                              {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                           .rank = 1,
                                           .dimensions = split_index_dim,
                                           .memType = QNN_TENSORMEMTYPE_RAW,
                                           .clientBuf = {.data = split_index.data(), .dataSize = static_cast<uint32_t>(((split_num_ - 1)) * sizeof(uint32_t))}}}}};

    vector<Qnn_Tensor_t> out = {};
    vector<vector<uint32_t>> outDims;
    vector<string *> outNames;
    auto outPutDataType = QNN_DATATYPE_FLOAT_32;
    auto quanDefined = QNN_DEFINITION_UNDEFINED;
    auto quantDecoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    float quantScale = 0.0000000000000000f;
    if (inputs[0]->dtype() == MLLM_TYPE_F16) {
        outPutDataType = QNN_DATATYPE_FLOAT_16;
        for (auto &output : outputs) {
            output->setDtype(MLLM_TYPE_F16);
        }

    } else if (inputs[0]->dtype() == MLLM_TYPE_I8) {
        for (auto &output : outputs) {
            output->setDtype(MLLM_TYPE_I8);
        }
        outPutDataType = QNN_DATATYPE_SFIXED_POINT_8;
        quanDefined = QNN_DEFINITION_DEFINED;
        quantDecoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        quantScale = inputs[0]->quant_param.scale;
        outputs[0]->quant_param = inputs[0]->quant_param;
    } else if (inputs[0]->dtype() == MLLM_TYPE_I16) {
        for (auto &output : outputs) {
            output->setDtype(MLLM_TYPE_I16);
        }
        outPutDataType = QNN_DATATYPE_SFIXED_POINT_16;
        quanDefined = QNN_DEFINITION_DEFINED;
        quantDecoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
        quantScale = inputs[0]->quant_param.scale;
        outputs[0]->quant_param = inputs[0]->quant_param;
    } else if (inputs[0]->dtype() == MLLM_TYPE_I32) {
        for (auto &output : outputs) {
            output->setDtype(MLLM_TYPE_I32);
        }
        outPutDataType = QNN_DATATYPE_SFIXED_POINT_32;
    }
    for (int i = 0; i < split_num_; i++) {
        outDims.push_back({static_cast<uint32_t>(outputs[i]->batch()),
                           static_cast<uint32_t>(outputs[i]->sequence()),
                           static_cast<uint32_t>(outputs[i]->head()),
                           static_cast<uint32_t>(outputs[i]->dimension())});
        outNames.push_back(new string(outputs[i]->name()));
        out.push_back((Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            .v1 = {
                .id = 0,
                .name = outNames[i]->c_str(),
                .type = getOutputTensorType(outputs[i]),
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = outPutDataType,
                .quantizeParams = {quanDefined,
                                   quantDecoding,
                                   {.scaleOffsetEncoding = {.scale = quantScale, .offset = 0}}},
                .rank = 4,
                .dimensions = outDims[i].data(),
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {.data = nullptr, .dataSize = 0}}});
    }

    return graphAddNode(name(), "Split", {inputs[0]->name()}, out, params);
}

} // namespace mllm
