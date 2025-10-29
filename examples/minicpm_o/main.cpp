#include <iostream>
#include <fmt/core.h>
#include "mllm/mllm.hpp"
#include "mllm/models/minicpm_o2_6/configuration_minicpmo.hpp"
// #include "mllm/models/minicpm_o2_6/modeling_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/modeling_resampler.hpp"
#include "mllm/models/minicpm_o2_6/modeling_siglip.hpp"
#include "mllm/models/minicpm_o2_6/tokenization_minicpmo.hpp"
#include "mllm/models/minicpm_o2_6/image_preprocessor_minicpmo.hpp"
#include "mllm/utils/AnyValue.hpp"
#include "mllm/preprocessor/visual/Image.hpp"

using mllm::Argparse;

MLLM_MAIN({
    mllm::Logger::level() = mllm::LogLevel::kError;

    auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
    auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
    auto& model_version = Argparse::add<std::string>("-mv|--model_version").help("Model version").required(true);
    auto& tokenizer_path = Argparse::add<std::string>("-t|--tokenizer_path").help("Tokenizer directory").required(true);
    auto& config_path = Argparse::add<std::string>("-c|--config_path").help("Config path").required(true);
    // RUN: ./main_minicpm_o -m ../../models/minicpm-o-2_6.mllm -mv v1 -t ../../tokenizer/MiniCPM-o-2_6/tokenizer.json -c ../../examples/minicpm_o/config_minicpm_o.json

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

       mllm::models::minicpmo::MiniCPMOMessage message;
       message.prompt = "现在你是太监，这个男子是皇上，你需要真心实意地奉承他";
       message.img_file_path = "/Users/kkkai/Desktop/pics.jpg";
       auto output = minicpmo_tokenizer.convertMessage(message);
       mllm::print(output["input_ids"].shape());
       mllm::print(output["pixel_values"].shape());
       mllm::print(output["tgt_sizes"].shape());
       mllm::print(output["image_bounds"].shape());

       auto param = mllm::load(model_path.get(), file_version);
       auto siglip = mllm::models::minicpmo::SiglipVisionModel("vpm", minicpmo_cfg);
       siglip.load(param);
       auto res = siglip(output["pixel_values"], output["tgt_sizes"])[0];
       auto resampler = mllm::models::minicpmo::Resampler("resampler", 64, 3584, 28, 1152);
       resampler.load(param);
       auto res2 = resampler(res, output["tgt_sizes"])[0];

        // auto minicpmo = mllm::models::minicpmo::MiniCPMOForCausalLM(minicpmo_cfg);

        // // Load model weights
        // auto param = mllm::load(model_path.get(), file_version);
        // minicpmo.load(param);

        // fmt::print("\n{:*^60}\n", " MiniCPM-o Interactive CLI ");
        // fmt::print("Enter 'exit' or 'quit' to end the session\n");
        // fmt::print("Supported modes: text, image+text, audio+text, multimodal\n\n");

        // while (true) {
        //     std::string mode;
        //     fmt::print("Mode (text/image/audio/multi) or 'exit': ");
        //     std::getline(std::cin, mode);

        //     if (mode == "exit" || mode == "quit") {
        //         break;
        //     }

        //     mllm::models::minicpmo::MiniCPMOInput input;

        //     // Handle different input modes
        //     if (mode == "image" || mode == "multi") {
        //         std::string image_path;
        //         fmt::print("Image path: ");
        //         std::getline(std::cin, image_path);
        //         if (!image_path.empty()) {
        //             input.img_file_path = image_path;
        //         }
        //     }

        //     if (mode == "audio" || mode == "multi") {
        //         std::string audio_path;
        //         fmt::print("Audio path: ");
        //         std::getline(std::cin, audio_path);
        //         if (!audio_path.empty()) {
        //             input.audio_file_path = audio_path;
        //         }
        //     }

        //     std::string prompt_text;
        //     fmt::print("Prompt text: ");
        //     std::getline(std::cin, prompt_text);
        //     input.prompt = prompt_text;

        //     try {
        //         fmt::print("Processing...\n");

        //         // Convert input to tokens
        //         auto input_tokens = minicpmo_tokenizer.convertMessage(input);

        //         // Process images if provided
        //         auto image_tensors = minicpmo_tokenizer.processImages(input);

        //         // Process audio if provided
        //         auto audio_tensors = minicpmo_tokenizer.processAudio(input);

        //         fmt::print("\nResponse: ");

        //         // TODO: Implement multimodal chat interface
        //         // For now, use text-only generation
        //         std::vector<int> token_ids;
        //         auto input_ptr = input_tokens.ptr<int>();
        //         auto seq_len = input_tokens.shape()[1];
        //         for (int i = 0; i < seq_len; ++i) {
        //             token_ids.push_back(input_ptr[i]);
        //         }

        //         // Generate response
        //         for (auto& step : minicpmo.chat(token_ids)) {
        //             auto token_str = minicpmo_tokenizer.detokenize(step.cur_token_id);
        //             std::wcout << token_str << std::flush;

        //             // TODO: Check for audio generation tokens
        //             if (minicpmo_tokenizer.isAudioToken(step.cur_token_id)) {
        //                 fmt::print("\n🔊 [Audio generation triggered - feature not implemented yet]\n");
        //             }
        //         }

        //         fmt::print("\n\n");

        //     } catch (const std::exception& e) {
        //         fmt::print(" Error: {}\n", e.what());
        //     }
        // }

        // fmt::print("Success！\n");
    }

#ifdef MLLM_PERFETTO_ENABLE
    mllm::perf::stop();
#endif

    mllm::shutdownContext();
    return 0;
})
