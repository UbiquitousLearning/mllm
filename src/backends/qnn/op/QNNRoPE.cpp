
#include "QNNRoPE.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>

namespace mllm {

vector<float> QNNRoPE::theta_;

vector<vector<float>> QNNRoPE::sin_;
vector<vector<float>> QNNRoPE::cos_;
int QNNRoPE::global_pose_type_ = -1;
int QNNRoPE::ishape_old;

extern void sinusoidal_position_embedding_llama(int seq_len, int output_dim, const vector<float> &theta,
                                                vector<vector<float>> &sin, vector<vector<float>> &cos, float attention_scaling = 1.0);
extern void sinusoidal_position_embedding_huggingface(int seq_len, int output_dim, const vector<float> &theta,
                                                      vector<vector<float>> &sin, vector<vector<float>> &cos, float attention_scaling = 1.0);
typedef float (*mllm_rope_init_func)(const OpParam &, std::vector<float> &);
extern unordered_map<RoPEThetaType, mllm_rope_init_func> rope_init_func_map;

QNNRoPE::QNNRoPE(Backend *bn, string opName, int pose_type) :
    QNNCommonOp(bn, opName) {
    pose_type_ = pose_type;

    sinTensor_.setBackend(bn);
    cosTensor_.setBackend(bn);
    hcntTensor_.setBackend(bn);

    scale_.setBackend(bn);
}

QNNRoPE::QNNRoPE(Backend *bn, string opName, int pose_type, float rope_theta, int max_position_embeddings) :
    QNNCommonOp(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    pos_max_ = max_position_embeddings;

    sinTensor_.setBackend(bn);
    cosTensor_.setBackend(bn);
    hcntTensor_.setBackend(bn);

    scale_.setBackend(bn);
}

QNNRoPE::QNNRoPE(Backend *bn, string opName, int pose_type, float rope_theta, float partial_rotary_factor, int max_position_embeddings) :
    QNNCommonOp(bn, opName) {
    pose_type_ = pose_type;
    rope_theta_ = rope_theta;
    partial_rotary_factor_ = partial_rotary_factor;
    pos_max_ = max_position_embeddings;

    sinTensor_.setBackend(bn);
    cosTensor_.setBackend(bn);
    hcntTensor_.setBackend(bn);

    scale_.setBackend(bn);
}

QNNRoPE::QNNRoPE(Backend *bn, string opName, OpParam &config) :
    QNNCommonOp(bn, opName) {
    config_ = config;
    pose_type_ = config.at("pose_type");
    auto it = config.find("rope_theta");
    if (it != config.end()) {
        rope_theta_ = it->second;
    }
    it = config.find("partial_rotary_factor");
    if (it != config.end()) {
        partial_rotary_factor_ = it->second;
    }
    it = config.find("max_position_embeddings");
    if (it != config.end()) {
        pos_max_ = it->second;
    }
    rope_type = (RoPEThetaType)config.at("rope_type");
}

ErrorCode QNNRoPE::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence(), inputs[0]->dimension());

    ishape = inputs[0]->dimension() * partial_rotary_factor_;

    // UINT8 input, FP32 ouput, only support dimension == 128
    assert(inputs[0]->dimension() == 128);

    return Op::reshape(inputs, outputs);
}

// from CPURoPE.cpp 2025/1/24
extern float _default_init_rope(const OpParam &config, vector<float> &theta);
// from CPURoPE.cpp 2025/1/24
extern float _compute_llama3_theta(const OpParam &config, vector<float> &theta);

ErrorCode QNNRoPE::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    // in case ishape is 0 when Op is the first one in the graph

    const unordered_map<RoPEThetaType, mllm_rope_init_func> rope_init_func_map = {
        {DEFAULT, _default_init_rope},
        {LLAMA3, _compute_llama3_theta},
    };

    if (sin_.empty() || ishape_old < ishape || global_pose_type_ != pose_type_) {
        auto calc_theta = rope_init_func_map.at(rope_type);
        auto config = config_;
        config["base"] = (float)rope_theta_;
        config["dim"] = ishape;
        float attention_scaling = calc_theta(config, theta_);

        global_pose_type_ = pose_type_;
        ishape_old = ishape;
        if (pose_type_ == LLAMAROPE) {
            sinusoidal_position_embedding_llama(pos_max_, ishape, theta_, sin_, cos_, attention_scaling);
        } else if (pose_type_ == PERSIMMONROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape / 2, theta_, sin_, cos_, attention_scaling);
        } else if (pose_type_ == HFHUBROPE || pose_type_ == MLAROPE) {
            sinusoidal_position_embedding_huggingface(pos_max_, ishape, theta_, sin_, cos_, attention_scaling);
        } else {
        }
    }

    float dequantScale = 0;
    dequantScale = scale_.hostPtr<float>()[0] / 127.0;
    dequantScale = roundf(dequantScale * 100000) / 100000;

    if (name().find("q_proj") != -1) {
        dequantScale = dequantScale / std::sqrt(outputs[0]->dimension());
    }

    auto type = QNN_DATATYPE_FLOAT_32;
    if (outputs[0]->dtype() == MLLM_TYPE_F32) {
        sinTensor_.setName(name() + ".sin");
        sinTensor_.reshape(1, 1, pos_max_, ishape / 2);
        sinTensor_.setDtype(MLLM_TYPE_F32);
        sinTensor_.alloc();

        cosTensor_.setName(name() + ".cos");
        cosTensor_.reshape(1, 1, pos_max_, ishape / 2);
        cosTensor_.setDtype(MLLM_TYPE_F32);
        cosTensor_.alloc();

        for (int i = 0; i < pos_max_; i++) {
            for (int j = 0; j < ishape / 2; j++) {
                sinTensor_.setDataAt<float>(0, 0, i, j, sin_[i][j] * dequantScale);
                cosTensor_.setDataAt<float>(0, 0, i, j, cos_[i][j] * dequantScale);
            }
        }

    } else if (outputs[0]->dtype() == MLLM_TYPE_F16) {
        sinTensor_.setName(name() + ".sin");
        sinTensor_.reshape(1, 1, pos_max_, ishape / 2);
        sinTensor_.setDtype(MLLM_TYPE_F32);
        sinTensor_.alloc();

        cosTensor_.setName(name() + ".cos");
        cosTensor_.reshape(1, 1, pos_max_, ishape / 2);
        cosTensor_.setDtype(MLLM_TYPE_F32);
        cosTensor_.alloc();

        for (int i = 0; i < pos_max_; i++) {
            for (int j = 0; j < ishape / 2; j++) {
                sinTensor_.setDataAt<float>(0, 0, i, j, static_cast<float>(sin_[i][j]));
                cosTensor_.setDataAt<float>(0, 0, i, j, static_cast<float>(cos_[i][j]));
            }
        }

        type = QNN_DATATYPE_FLOAT_16;
    }

    uint32_t sin_dimensions[] = {static_cast<uint32_t>(pos_max_), static_cast<uint32_t>(ishape / 2)};
    uint32_t cos_dimensions[] = {static_cast<uint32_t>(pos_max_), static_cast<uint32_t>(ishape / 2)};

    auto sinWeightsName = name() + ".sin.weights";

    qnnBackend_->modelAddTensor(sinWeightsName, // Node Name
                                (Qnn_Tensor_t){
                                    .version = QNN_TENSOR_VERSION_1,
                                    .v1 = {
                                        .id = 0,
                                        .name = sinWeightsName.c_str(),
                                        .type = QNN_TENSOR_TYPE_STATIC,
                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                        .dataType = QNN_DATATYPE_FLOAT_32,
                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                        .rank = 2,
                                        .dimensions = sin_dimensions,
                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                        .clientBuf = {.data = sinTensor_.hostPtr<uint8_t>(),
                                                      .dataSize = static_cast<uint32_t>(sinTensor_.cntSize())}}});
    // free sin_
    // sin_.free();

    auto cosWeightsName = name() + ".cos.weights";
    qnnBackend_->modelAddTensor(cosWeightsName, // Node Name
                                (Qnn_Tensor_t){
                                    .version = QNN_TENSOR_VERSION_1,
                                    .v1 = {

                                        .id = 0,
                                        .name = cosWeightsName.c_str(),
                                        .type = QNN_TENSOR_TYPE_STATIC,
                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                        .dataType = QNN_DATATYPE_FLOAT_32,
                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                        .rank = 2,
                                        .dimensions = cos_dimensions,
                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                        .clientBuf = {.data = cosTensor_.hostPtr<uint8_t>(),
                                                      .dataSize = static_cast<uint32_t>(cosTensor_.cntSize())}}});
    // free cos_
    // cos_.free();

    auto hcntName = name() + ".h_cnt";
    uint32_t hcntDimension[1] = {1};
    qnnBackend_->modelAddTensor(hcntName, // Node Name
                                (Qnn_Tensor_t){
                                    .version = QNN_TENSOR_VERSION_1,
                                    .v1 = {
                                        .id = 0,
                                        .name = hcntName.c_str(),
                                        .type = QNN_TENSOR_TYPE_APP_WRITE,
                                        .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                                        .dataType = QNN_DATATYPE_UINT_32,
                                        .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                                           QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                                           {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                                        .rank = 1,
                                        .dimensions = hcntDimension,
                                        .memType = QNN_TENSORMEMTYPE_RAW,
                                        .clientBuf = {.data = nullptr,
                                                      .dataSize = 0}}});

    qnnBackend_->pushInputBuffers(hcntTensor_.hostPtr<uint8_t>());

    vector<Qnn_Param_t> params_rope = {
        {.paramType = QNN_PARAMTYPE_SCALAR,
         .name = "pose_type",
         .scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = static_cast<uint32_t>(global_pose_type_)}}}};

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
                .dataType = type,
                .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                   QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                   {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                .rank = 4,
                .dimensions = dimOut,
                .memType = QNN_TENSORMEMTYPE_RAW,
                .clientBuf = {.data = nullptr,
                              .dataSize = 0}}}};

    return graphAddNode(name(), "RoPE", {inputs[0]->name(), sinWeightsName, cosWeightsName, hcntName}, out, params_rope, "LLaMAPackage");
}

ErrorCode QNNRoPE::load(AbstructLoader &loader) {
    hcntTensor_.setName(name() + ".hcnt.tensor");
    hcntTensor_.reshape(1, 1, 1, 1);
    hcntTensor_.setDtype(MLLM_TYPE_I32);
    hcntTensor_.alloc();

    string scaleName = name();
    string scaleTypeName = "output_scale";

    std::string wordToRemove = "rope";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());
    }

    scale_.setName(scaleName + scaleTypeName);
    scale_.reshape(1, 1, 1, 1);
    scale_.setDtype(MLLM_TYPE_F32);
    scale_.alloc();
    loader.load(&scale_);

    return Op::load(loader);
}

ErrorCode QNNRoPE::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode QNNRoPE::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    h_cnt_ += inputs[0]->sequence();
    hcntTensor_.setDataAt(0, 0, 0, 0, h_cnt_);

    return QNNCommonOp::execute(inputs, outputs);
}

} // namespace mllm
