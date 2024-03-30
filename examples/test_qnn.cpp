/***
* NOTE add target_compile_definitions(test_qnn PRIVATE USE_QNN) in CMakeLists.txt
*/

#include "Types.hpp"
#include "cmdline.h"
#include "models/opt/modeling_opt_qnn.hpp"


using namespace mllm;

Tensor get_input_qnn() {
    Tensor tensor1(1, 1, 1, 1, Module::backends[MLLM_QNN], true);
    tensor1.setName(std::move("input"));
    tensor1.status() = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    return tensor1;
}

int main(int argc, char **argv) {
    Module::initBackend(MLLM_CPU);
    cmdline::parser cmdParser;
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/XXXX.mllm");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string model_path = cmdParser.get<string>("model");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto nameconfig =  OPTNameConfig();
    nameconfig.init(HFHUBROPE);
    auto model = QNNEncoderBlock(1, 1, 1, 100, nameconfig, "name.");

    auto result = model({get_input_qnn()});
}