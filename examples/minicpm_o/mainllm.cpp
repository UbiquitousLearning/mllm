#include <iostream>
#include <fmt/core.h>
#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_resampler.hpp"
#include "mllm/models/minicpm_o2_6/modeling_siglip.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/image_preprocessor_minicpmo.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/preprocessor/visual/Image.hpp"
#include "cnpy.h"

using mllm::Argparse;

MLLM_MAIN({

    mllm::Logger::level() = mllm::LogLevel::kError;
    //mllm::setPrintMaxElementsPerDim(1000);
    
    auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
    auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
    auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
    auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
    auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
    // RUN: ./main_minicpm_o2 -m ../../models/minicpm-o-2_6.mllm -mv v1 -t ../../tokenizer/MiniCPM-o-2_6/tokenizer.json -c ../../examples/minicpm_o/config_minicpm_o.json

    Argparse::parse(argc, argv);

#ifdef MLLM_PERFETTO_ENABLE
    mllm::perf::start();
#endif

    mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;  
    if (model_version.get() == "v1") {
        file_version = mllm::ModelFileVersion::kV1;
    } else if (model_version.get() == "v2") {
        file_version = mllm::ModelFileVersion::kV2;
    }

    if (help.isSet()) {
        Argparse::printHelp();
        mllm::shutdownContext();
        return 0;
    }
    {
      auto minicpmo_cfg = mllm::models::minicpmo::MiniCPMOConfig(config_path.get());
      auto minicpmo_tokenizer = mllm::models::minicpmo::MiniCPMOTokenizer(tokenizer_path.get());
      auto minicpmo = mllm::models::minicpmo::MiniCPMOForCausalLM(minicpmo_cfg);

      auto param = mllm::load(model_path.get(), file_version);
      minicpmo.llm_.llm.load(param);
      minicpmo.vpm_.load(param);
      minicpmo.resampler_.load(param);
      //minicpmo.audio_proj_.load(param);
      //minicpmo.tts_proj_.load(param);

      fmt::print("\n{:*^60}\n", " MiniCPM-o Interactive CLI ");
      fmt::print("Enter 'exit' or 'quit' to end the session\n");

      std::string image_path = "/Users/kkkai/Desktop/pics.jpg";
      std::string prompt_text = "描述图片中物体";
      mllm::models::minicpmo::MiniCPMOMessage message;
      message.prompt = prompt_text;
      message.img_file_path = image_path;

      // fmt::print("📷 Image path (or 'exit/quit'): ");
      // std::getline(std::cin, image_path);
      // if (image_path == "exit" || image_path == "quit") { return 0; }
      // fmt::print("💬 Prompt text: ");
      // std::getline(std::cin, prompt_text);

      fmt::print("Processing...\n");
      auto inputs = minicpmo_tokenizer.convertMessage(message);

      fmt::print("\nResponse: ");

      int token_count = 0;
      for(auto& step : minicpmo.chat(inputs)){
        auto token_str = minicpmo_tokenizer.detokenize(step.cur_token_id);
        std::wcout<< token_str << std::flush;

        token_count++;
        if(token_count >= 50) break; // Limit output for debugging
      }

      fmt::print("\n{}\n", std::string(60, '-'));
      


#ifdef MLLM_PERFETTO_ENABLE
    mllm::perf::stop();
    mllm::perf::saveReport("minicpmo.perf");
#endif

    mllm::memoryReport();
    mllm::shutdownContext();
    return 0;
    }
})
