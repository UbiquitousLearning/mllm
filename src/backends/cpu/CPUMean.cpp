
#include "CPUMean.hpp"

namespace mllm {

CPUMean::CPUMean(Backend *bn,  string opName, int axis, int threadCount) : thread_count(threadCount),
    Op(bn, opName) {
    axis_ = (Chl)axis;
}

ErrorCode CPUMean::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    int batch = inputs[0]->batch();
    int head = inputs[0]->head();
    int sequence = inputs[0]->sequence();
    int dimension = inputs[0]->dimension();
    switch (axis_) {
    case BATCH:
        batch = 1;
        break;
    case HEAD:
        head = 1;
        break;
    case SEQUENCE:
        sequence = 1;
        break;
    case DIMENSION:
        dimension = 1;
        break;
    default:
        break;
    }
    outputs[0]->reshape(batch, head, sequence, dimension);
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUMean::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    auto input = inputs[0];
    auto output = outputs[0];
    int batch = input->batch();
    int dim = input->dimension();
    int seq = input->sequence();
    int head = input->head();

    switch (axis_) {
    case BATCH: {
        for (int h = 0; h < head; h++) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < dim; d++) {
                    float sum = 0.0f;
                    for (int n = 0; n < batch; n++) {
                        sum += inputs[0]->dataAt<float>(n, h, s, d);
                    }
                    outputs[0]->setDataAt<float>(0, h, s, d, sum / seq);
                }
            }
        }
        break;
    }
    case HEAD: {
        for (int n = 0; n < batch; n++) {
            for (int s = 0; s < seq; ++s) {
                for (int d = 0; d < dim; d++) {
                    float sum = 0.0f;
                    for (int h = 0; h < head; h++) {
                        sum += inputs[0]->dataAt<float>(n, h, s, d);
                    }
                    outputs[0]->setDataAt<float>(n, 0, s, d, sum / seq);
                }
            }
        }
        break;
    }
    case SEQUENCE:{
        for (int n = 0; n < batch; n++) {
            for (int h = 0; h < head; h++) {
                for (int d = 0; d < dim; d++) {
                    float sum = 0.0f;
                    for (int s = 0; s < seq; ++s) {
                        sum += inputs[0]->dataAt<float>(n, h, s, d);
                    }
                    outputs[0]->setDataAt<float>(n, h, 0, d, sum / seq);
                }
            }
        }
        break;
    }
    case DIMENSION:{
        for (int n = 0; n < batch; n++) {
            for (int h = 0; h < head; h++) {
                for (int s = 0; s < seq; s++) {
                    float sum = 0.0f;
                    for (int d = 0; d < inputs[0]->dimension(); ++d) {
                        sum += inputs[0]->dataAt<float>(n, h, s, d);
                    }
                    outputs[0]->setDataAt<float>(n, h, s, 0, sum / inputs[0]->dimension());
                }
            }
        }
        break;
    }
    default:
        break;
    }
    return Op::execute(inputs, outputs);
}
} // namespace mllm

