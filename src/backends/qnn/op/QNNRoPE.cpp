
#include "QNNRoPE.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {
QNNRoPE::QNNRoPE(Backend *bn, string opName, int pose_type) :
    QNNCommonOp(bn, opName), pose_type_(pose_type) {
    cos_.setBackend(bn);
    sin_.setBackend(bn);
}

ErrorCode QNNRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    ishape = inputs[0]->dimension();

    return Op::reshape(inputs, outputs);
}

ErrorCode QNNRoPE::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // in case ishape is 0 when Op is the first one in the graph
    if (!sin_.allocted()) {
        if (pose_type_ == HFHUBROPE) {
            sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape, sin_, cos_);
        } else if (pose_type_ == LLAMAROPE) {
            sinusoidal_position_embedding(1, 1, pos_max_, ishape, sin_, cos_);
        } else if (pose_type_ == PERSIMMONROPE) {
            sinusoidal_position_embedding_hf(1, 1, pos_max_, ishape / 2, sin_, cos_);
        } else {
        }
    }

    uint32_t sin_dimensions[] = {static_cast<uint32_t>(pos_max_), static_cast<uint32_t>(ishape)};
    uint32_t cos_dimensions[] = {static_cast<uint32_t>(pos_max_), static_cast<uint32_t>(ishape)};

    qnnBackend_->modelAddTensor(sin_.name(), // Node Name
                                (Qnn_Tensor_t){
                                    .version = QNN_TENSOR_VERSION_1,
                                    .v1 = {
                                        .id = 0,
                                        .name = sin_.name().c_str(),
                                        .type = QNN_TENSOR_TYPE_STATIC,
                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                        .dataType = QNN_DATATYPE_FLOAT_32,
                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                        .rank = 2,
                                        .dimensions = sin_dimensions,
                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                        .clientBuf = {.data = sin_.hostPtr<void>(),
                                                      .dataSize = static_cast<uint32_t>(sin_.cntSize())}}});
    // free sin_
    // sin_.free();

    qnnBackend_->modelAddTensor(cos_.name(), // Node Name
                                (Qnn_Tensor_t){
                                    .version = QNN_TENSOR_VERSION_1,
                                    .v1 = {
                                        .id = 0,
                                        .name = cos_.name().c_str(),
                                        .type = QNN_TENSOR_TYPE_STATIC,
                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                        .dataType = QNN_DATATYPE_FLOAT_32,
                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                        .rank = 2,
                                        .dimensions = cos_dimensions,
                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                        .clientBuf = {.data = cos_.hostPtr<void>(),
                                                      .dataSize = static_cast<uint32_t>(cos_.cntSize())}}});
    // free cos_
    // cos_.free();

    uint32_t pose_type = 2;
    vector<Qnn_Param_t> params_rope = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "pose_type",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = pose_type}}}};

    uint32_t dimOut[4] = {static_cast<uint32_t>(inputs[0]->batch()),
                          static_cast<uint32_t>(inputs[0]->sequence()),
                          static_cast<uint32_t>(inputs[0]->head()),
                          static_cast<uint32_t>(inputs[0]->dimension())};
    auto outName = outputs[0]->name();
    vector<Qnn_Tensor_t> out = {
        (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            .v1 = {
                .id = 0,
                .name = outName.c_str(),
                .type = getOutputTensorType(outputs[0]),
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = QNN_DATATYPE_FLOAT_32,
                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                .rank = 4,
                .dimensions = dimOut,
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {.data = nullptr,
                              .dataSize = 0}}}};
    return graphAddNode(name(), "RoPE", {inputs[0]->name(), sin_.name(), cos_.name()}, out, params_rope, "LLaMAPackage");
}

ErrorCode QNNRoPE::load(AbstructLoader &loader) {
    sin_.setName(name() + ".sin");
    cos_.setName(name() + ".cos");

    return Op::load(loader);
}

ErrorCode QNNRoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

void QNNRoPE::sinusoidal_position_embedding(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape(batch_size, nums_head, seq_len, output_dim);
    cos.reshape(batch_size, nums_head, seq_len, output_dim);
    sin.setDtype(MLLM_TYPE_F32);
    cos.setDtype(MLLM_TYPE_F32);
    sin.alloc();
    cos.alloc();
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < nums_head; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < output_dim; d += 2) {
                    int i = (int)d / 2;
                    float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
                    float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));
                    sin.setDataAt<float>(n, h, s, d, sin_value);
                    cos.setDataAt<float>(n, h, s, d, cos_value);
                    if (d + 1 < output_dim) {
                        sin.setDataAt<float>(n, h, s, d + 1, sin_value);
                        cos.setDataAt<float>(n, h, s, d + 1, cos_value);
                    }
                }
            }
        }
    }
}

void QNNRoPE::sinusoidal_position_embedding_hf(int batch_size, int nums_head, int seq_len, int output_dim, Tensor &sin, Tensor &cos) {
    sin.reshape(batch_size, nums_head, seq_len, output_dim);
    cos.reshape(batch_size, nums_head, seq_len, output_dim);
    sin.setDtype(MLLM_TYPE_F32);
    cos.setDtype(MLLM_TYPE_F32);
    sin.alloc();
    cos.alloc();
    for (int n = 0; n < batch_size; ++n) {
        for (int h = 0; h < nums_head; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                for (int d = 0; d < output_dim; d += 2) {
                    int i = (int)d;
                    if (d >= (int)output_dim / 2) {
                        i = (int)(d - output_dim / 2);
                    }
                    float sin_value = std::sin(s / std::pow(10000, 2.0 * i / output_dim));
                    float cos_value = std::cos(s / std::pow(10000, 2.0 * i / output_dim));

                    sin.setDataAt<float>(n, h, s, d, sin_value);
                    cos.setDataAt<float>(n, h, s, d, cos_value);
                    if (d + 1 < output_dim) {
                        sin.setDataAt<float>(n, h, s, d + 1, sin_value);
                        cos.setDataAt<float>(n, h, s, d + 1, cos_value);
                    }
                }
            }
        }
    }
}

ErrorCode QNNRoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // TODO: h_cnt_ as an input to ROPE.
    h_cnt_ += inputs[0]->sequence();

    return QNNCommonOp::execute(inputs, outputs);
}

} // namespace mllm
