#include <iostream>
#include <string>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include <mllm/models/smollm3_3B/modeling_smollm3.hpp>
#include <mllm/models/smollm3_3B/tokenization_smollm3.hpp>
#include <mllm/models/smollm3_3B/configuration_smollm3.hpp>

using mllm::Argparse;

MLLM_MAIN({
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& model_path = Argparse::add<std::string>("-m|--model_path").help("Model path").required(true);
  auto& think_mode = Argparse::add<bool>("--think").help("Enable thinking mode");
  
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  try {
    std::string actual_model_path = model_path.get();
    std::string actual_config_path = actual_model_path + "/config.json";
    std::string actual_tokenizer_path = actual_model_path + "/tokenizer.json";  // 统一使用基于model_path的路径

    auto cfg = mllm::models::smollm3::Smollm3Config(actual_config_path);
    auto tokenizer = mllm::models::smollm3::SmolLM3Tokenizer(actual_tokenizer_path);
    auto model = mllm::models::smollm3::Smollm3ForCausalLM(cfg);

    // Load model parameters
    auto param = mllm::load(actual_model_path + "/model.mllm", mllm::ModelFileVersion::kV2);
    model.load(param);

    fmt::print("\n{:*^60}\n", " SmolLM3-3B Interactive CLI ");
    fmt::print("Think mode: {}\n", think_mode.isSet() ? "ENABLED" : "DISABLED");
    fmt::print("Enter 'exit' or 'quit' to end the session\n");
    fmt::print("Enter 'clear' to clear conversation history\n");
    fmt::print("Enter 'reset' to reset model state\n\n");

    while (true) {
      std::string prompt_text;
      fmt::print("💬 Prompt: ");
      std::getline(std::cin, prompt_text);

      if (prompt_text == "exit" || prompt_text == "quit") {
        break;
      }

      if (prompt_text == "clear") {
        fmt::print("🗑️  Conversation history cleared.\n\n");
        continue;
      }

      if (prompt_text == "reset") {
        model.resetCache();
        fmt::print("🔄 Model state reset.\n\n");
        continue;
      }

      if (prompt_text.empty()) {
        continue;
      }

      try {
        // Prepare input message
        mllm::models::smollm3::SmolLM3Message message;
        message.prompt = prompt_text;
        message.enable_thinking = think_mode.isSet();

        // Use applyChatTemplate and encode to build input
        std::string input_text = tokenizer.applyChatTemplate(message.prompt, message.enable_thinking);
        auto input_ids = tokenizer.encode(input_text);
        
        // Create input tensor using fromVector
        auto sequence_tensor = mllm::Tensor::fromVector(input_ids, {1, static_cast<int>(input_ids.size())}, mllm::kInt64);
        
        mllm::models::ARGenerationOutputPast inputs;
        inputs["sequence"] = sequence_tensor;

        fmt::print("🤖 SmolLM3: ");

        // Stream generation response
        std::string full_response;
        int token_count = 0;
        bool in_thinking = think_mode.isSet();
        bool thinking_completed = false;
        bool response_started = false;
        
        for (auto& step : model.chat(inputs)) {
          // Convert single token id to vector, then decode
          std::vector<int64_t> token_vec = {step.cur_token_id};
          std::string token_text = tokenizer.decode(token_vec);
          
          // Think mode processing
          if (in_thinking && !thinking_completed) {
            // Check if think end marker is encountered
            if (token_text.find("</think>") != std::string::npos) {
              thinking_completed = true;
              in_thinking = false;
              fmt::print("\n🤖 Response: ");
              response_started = true;
              continue;
            }
            
            // Display thinking content in think mode
            if (think_mode.isSet()) {
              std::cout << token_text << std::flush;
            }
            continue;
          }
          
          // Normal response output
          if (!response_started && !think_mode.isSet()) {
            response_started = true;
          }
          
          std::cout << token_text << std::flush;
          full_response += token_text;
          token_count++;
          
          // Check if end token is generated
          if (step.cur_token_id == cfg.eos_token_id) {
            break;
          }
          
          // Safety limit
          if (token_count > 512) {
            fmt::print("\n[Output truncated]");
            break;
          }
        }

        fmt::print("\n{}\n", std::string(60, '-'));

        // Reset cache
        model.resetCache();

      } catch (const std::exception& e) {
        fmt::print("\n❌ Error: {}\n{}\n", e.what(), std::string(60, '-'));
        model.resetCache();
      }
    }

    fmt::print("\n👋 Thank you for using SmolLM3-3B!\n");

  } catch (const std::exception& e) {
    fmt::print("\n❌ Error: {}\n", e.what());
    return -1;
  }

  return 0;
})
