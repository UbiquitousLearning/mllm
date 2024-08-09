
#include "QNNSplitInput.hpp"
#include "QnnTypes.h"
#include "Types.hpp"
#include "QNNCommonOp.hpp"
#include <cstdint>
#include <memory>

namespace mllm {
QNNSplitInput::QNNSplitInput(Backend *bn, string opName, bool isPrompt, int num) :
    QNNCommonOp(bn, opName) {
    
    isPrompt_ = isPrompt;
    num_ = num;
    scale1_.setBackend(bn);
    scale2_.setBackend(bn);

    residual_.setBackend(bn);
}

ErrorCode QNNSplitInput::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    assert(inputs.size() == 1);
    assert(outputs.size() == 2 || outputs.size() == 1);

    if (isPrompt_) {

        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 5, inputs[0]->dimension());
        if (num_ == 2)
            outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 5, inputs[0]->dimension());

    } else {

        outputs[0]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 2, inputs[0]->dimension());
        outputs[1]->reshape(inputs[0]->batch(), inputs[0]->head(), inputs[0]->sequence() / 2, inputs[0]->dimension());

    }
    

    return Op::reshape(inputs, outputs);
}

// ErrorCode QNNSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {  

//     seqs_.setName(name() + ".seqs");
//     seqs_.reshape(1, 1, 1, outputs.size());
//     seqs_.setDtype(MLLM_TYPE_I32);
//     seqs_.setBackend(qnnBackend_);
//     seqs_.alloc();

//     seqs_.setDataAt<uint32_t>(0, 0, 0, 0, outputs[0]->sequence());
//     seqs_.setDataAt<uint32_t>(0, 0, 0, 1, outputs[1]->sequence());

//     vector<Qnn_Param_t> paramsSplit = {
//         {.paramType = QNN_PARAMTYPE_SCALAR,
//          .name = "num",
//          {.scalarParam = (Qnn_Scalar_t){QNN_DATATYPE_UINT_32, {.uint32Value = static_cast<uint32_t>(outputs.size())}}}},
//     };

//     uint32_t dimensionsSeqs[1] = {2};
//     qnnBackend_->modelAddTensor(seqs_.name(), (Qnn_Tensor_t){
//                                                      .version = QNN_TENSOR_VERSION_1,
//                                                      {.v1 = {
//                                                           .id = 0,
//                                                           .name = seqs_.name().c_str(),
//                                                           .type = QNN_TENSOR_TYPE_STATIC,
//                                                           .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                                                           .dataType = QNN_DATATYPE_UINT_32,
//                                                           .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                                                              QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                                                              {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                                                           .rank = 1,
//                                                           .dimensions = dimensionsSeqs,
//                                                           .memType = QNN_TENSORMEMTYPE_RAW,
//                                                           {.clientBuf = {.data = seqs_.hostPtr<uint8_t>(),
//                                                                          .dataSize = 4 * 2}}}}});
//     auto data_type = QNN_DATATYPE_FLOAT_32;
//     if (outputs[0]->dtype() == MLLM_TYPE_I8) {
//         std::cout << "QNN splitinput INT8 op" << std::endl;
//         data_type = QNN_DATATYPE_SFIXED_POINT_8;
//     }

//     inputs[0]->printShape();
//     outputs[0]->printShape();
//     outputs[1]->printShape();

//     auto outName_0 = outputs[0]->name();
//     auto outName_1 = outputs[1]->name();
//     uint32_t dimensionsOut_0[4] = {static_cast<uint32_t>(outputs[0]->batch()),
//                                    static_cast<uint32_t>(outputs[0]->sequence()),
//                                    static_cast<uint32_t>(outputs[0]->head()),
//                                    static_cast<uint32_t>(outputs[0]->dimension())};

//     uint32_t dimensionsOut_1[4] = {static_cast<uint32_t>(outputs[1]->batch()),
//                                     static_cast<uint32_t>(outputs[1]->sequence()),
//                                     static_cast<uint32_t>(outputs[1]->head()),
//                                     static_cast<uint32_t>(outputs[1]->dimension())};



//     float quantScale1 = 0;
//     quantScale1 = scale1_.hostPtr<float>()[0]  / 127.0;
//     quantScale1 = roundf(quantScale1 * 10000) / 10000;


//     vector<Qnn_Tensor_t> activation_outputs = {
//                                             (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
//                                             {.v1 = {
//                                                  .id = 0,
//                                                  .name = outName_0.c_str(),
//                                                  .type = getOutputTensorType(outputs[0]),
//                                                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                                                  .dataType = data_type,
//                                                  .quantizeParams = {QNN_DEFINITION_DEFINED,
//                                                                     QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
//                                                                     {.scaleOffsetEncoding = {.scale = quantScale1, .offset = 0}}},
//                                                  .rank = 4,
//                                                  .dimensions = dimensionsOut_0,
//                                                  .memType = QNN_TENSORMEMTYPE_RAW,
//                                                  {.clientBuf = {.data = nullptr,
//                                                                 .dataSize = 0}}}}},            
//                                             (Qnn_Tensor_t){QNN_TENSOR_VERSION_1,
//                                             {.v1 = {
//                                                  .id = 1,
//                                                  .name = outName_1.c_str(),
//                                                  .type = getOutputTensorType(outputs[1]),
//                                                  .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
//                                                  .dataType = QNN_DATATYPE_FLOAT_32,
//                                                  .quantizeParams = {QNN_DEFINITION_UNDEFINED,
//                                                                     QNN_QUANTIZATION_ENCODING_UNDEFINED,
//                                                                     {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
//                                                  .rank = 4,
//                                                  .dimensions = dimensionsOut_1,
//                                                  .memType = QNN_TENSORMEMTYPE_RAW,
//                                                  {.clientBuf = {.data = nullptr,
//                                                                 .dataSize = 0}}}}}};

//     return graphAddNode(name() + ".split", "SplitInput", {inputs[0]->name(), seqs_.name()}, activation_outputs, paramsSplit, "LLaMAPackage");
// }

ErrorCode QNNSplitInput::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {  
    // we add input tensor in setup.
    
    uint32_t dimensionsOut_0[4] = {static_cast<uint32_t>(outputs[0]->batch()),
                                    static_cast<uint32_t>(outputs[0]->sequence()),
                                    static_cast<uint32_t>(outputs[0]->head()),
                                    static_cast<uint32_t>(outputs[0]->dimension())};

    float quantScale1 = 0;
    quantScale1 = scale1_.hostPtr<float>()[0]  / 127.0;
    quantScale1 = roundf(quantScale1 * 100000) / 100000;

    qnnBackend_->modelAddTensor(inputs[0]->name().c_str(), (Qnn_Tensor_t){
        .version = QNN_TENSOR_VERSION_1,
        {.v1 = {
                .id = 0,
                .name = outputs[0]->name().c_str(),
                .type = QNN_TENSOR_TYPE_APP_WRITE,
                .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                .dataType = QNN_DATATYPE_SFIXED_POINT_8,
                .quantizeParams = {QNN_DEFINITION_DEFINED,
                                    QNN_QUANTIZATION_ENCODING_SCALE_OFFSET,
                                    {.scaleOffsetEncoding = {.scale = quantScale1, .offset = 0}}},
                .rank = 4,
                .dimensions = dimensionsOut_0,
                .memType = QNN_TENSORMEMTYPE_RAW,
                {.clientBuf = {.data = nullptr,
                            .dataSize = 0}}}}});

    if (num_ == 2) {

        outputs[1]->setDtype(MLLM_TYPE_F32);
        outputs[1]->alloc();

        uint32_t dimensionsOut_1[4] = {static_cast<uint32_t>(outputs[1]->batch()),
                                static_cast<uint32_t>(outputs[1]->sequence()),
                                static_cast<uint32_t>(outputs[1]->head()),
                                static_cast<uint32_t>(outputs[1]->dimension())};
                                
        qnnBackend_->modelAddTensor(inputs[0]->name().c_str(), (Qnn_Tensor_t){
            .version = QNN_TENSOR_VERSION_1,
            {.v1 = {
                    .id = 0,
                    .name = outputs[1]->name().c_str(),
                    .type = QNN_TENSOR_TYPE_APP_WRITE,
                    .dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
                    .dataType = QNN_DATATYPE_FLOAT_32,
                    .quantizeParams = {QNN_DEFINITION_UNDEFINED,
                                    QNN_QUANTIZATION_ENCODING_UNDEFINED,
                                    {.scaleOffsetEncoding = {.scale = 0.0000000000000000f, .offset = 0}}},
                    .rank = 4,
                    .dimensions = dimensionsOut_0,
                    .memType = QNN_TENSORMEMTYPE_RAW,
                    {.clientBuf = {.data = nullptr,
                                .dataSize = 0}}}}});

        qnnBackend_->pushInputBuffers(outputs[1]->hostPtr<uint8_t>());

    }

    return MLLM_NO_ERROR;
    
}

ErrorCode QNNSplitInput::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    
    seqs_.free();

    return MLLM_NO_ERROR;
}

ErrorCode QNNSplitInput::load(AbstructLoader &loader) {
    string scaleName = name();

    std::string wordToRemove = "or_split";
    int pos = scaleName.find(wordToRemove);
    if (pos != -1) {
        scaleName.erase(pos, wordToRemove.length());

        // o
        scale1_.setName(scaleName + "o_proj.input_scale");
        scale1_.reshape(1, 1, 1, 1);
        scale1_.setDtype(MLLM_TYPE_F32);
        scale1_.alloc();
        loader.load(&scale1_);

    } else if (scaleName.find("ires_split") != -1) {

        pos = scaleName.find("ires_split");
        wordToRemove = "ires_split";
        scaleName.erase(pos, wordToRemove.length());

        // q
        scale1_.setName(scaleName + "q_proj.input_scale");
        scale1_.reshape(1, 1, 1, 1);
        scale1_.setDtype(MLLM_TYPE_F32);
        scale1_.alloc();
        loader.load(&scale1_);

    } else if (scaleName.find("fres_split") != -1) {

        pos = scaleName.find("fres_split");
        wordToRemove = "fres_split";
        scaleName.erase(pos, wordToRemove.length());

        // fc1
        scale1_.setName(scaleName + "up_proj.input_scale");
        scale1_.reshape(1, 1, 1, 1);
        scale1_.setDtype(MLLM_TYPE_F32);
        scale1_.alloc();
        loader.load(&scale1_);

    } else {
        exit(-1);
    }

    return Op::load(loader);
}


ErrorCode QNNSplitInput::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {

    memcpy(outputs[1]->hostPtr<uint8_t>(), inputs[0]->hostPtr<uint8_t>() + outputs[0]->cntSize(), outputs[1]->cntSize());

    return MLLM_NO_ERROR;
}

} // namespace mllm
