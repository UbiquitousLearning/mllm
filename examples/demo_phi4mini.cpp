#include <iostream>
#include "cmdline.h"
#include "models/phi4mini/modeling_phi4.hpp"
#include "models/phi4mini/tokenization_phi4mini.hpp"
#include "processor/PostProcess.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;  
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false,
                          "/data/lyw/phi4-mini/phi4_vocab.mllm"); 
    cmdParser.add<string>("model", 'm', "specify mllm model path", false,
                          "/data/lyw/phi4-mini/phi4-mini.mllm"); 
    cmdParser.add<int>("limits", 'l', "max KV cache size", false, 6000);
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string merges_path = "/data/lyw/phi4-mini/merges.txt";
    string model_path = cmdParser.get<string>("model");
    int tokens_limit = cmdParser.get<int>("limits");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto tokenizer = Phi4Tokenizer(vocab_path, merges_path, false);
    
    Phi4Config config(
        tokens_limit, 
        "4-mini",     
        HFHUBROPE,    
        200064        
    );
    auto model = Phi4Model(config);
    model.load(model_path);

    vector<string> in_strs = {
        "who are you?",
        "What can you do?",
        "Please introduce Beijing University of Posts and Telecommunications."};

    for (int i = 0; i < in_strs.size(); ++i) {
        auto in_str_origin = in_strs[i];  
        auto in_str = tokenizer.apply_chat_template(in_str_origin);
        auto input_tensor = tokenizer.tokenize(in_str);

        std::cout << std::endl;
        std::cout << "[Q] " << in_str_origin << std::endl;
        std::cout << "[A] " << std::flush;

        for (int step = 0; step < 100; ++step) {
            auto result = model({input_tensor});
            auto [out_string, out_token] = tokenizer.detokenize(result[0]);
            auto [not_end, output_string] = tokenizer.postprocess(out_string);
            if (!not_end) { break; }
            std::cout << output_string << std::flush;
            chatPostProcessing(out_token, input_tensor, {});
        }
        printf("\n");
        model.clear_kvcache();
        model.profiling();
    }

    return 0;
}