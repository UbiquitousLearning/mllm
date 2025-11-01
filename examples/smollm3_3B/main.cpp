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
  auto& debug_mode = Argparse::add<bool>("--debug").help("Enable debug output");
  
  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  try {
    std::string actual_model_path = model_path.get();
    std::string actual_config_path = actual_model_path + "/config.json";
    std::string actual_tokenizer_path = actual_model_path + "/tokenizer.json";

    if (debug_mode.isSet()) {
      fmt::print("Loading config from: {}\n", actual_config_path);
    }
    auto cfg = mllm::models::smollm3::Smollm3Config(actual_config_path);
    
    if (debug_mode.isSet()) {
      fmt::print("Loading tokenizer from: {}\n", actual_tokenizer_path);
    }
    auto tokenizer = mllm::models::smollm3::SmolLM3Tokenizer(actual_tokenizer_path);
    
    if (debug_mode.isSet()) {
      fmt::print("Creating model...\n");
    }
    auto model = mllm::models::smollm3::Smollm3ForCausalLM(cfg);

    if (debug_mode.isSet()) {
      fmt::print("Loading model weights from: {}\n", actual_model_path);
    }
    
    // 加载模型参数
    auto param = mllm::load(actual_model_path + "/model.mllm", mllm::ModelFileVersion::kV2);
    model.load(param);

    fmt::print("\n{:*^60}\n", " SmolLM3-3B Interactive CLI ");
    fmt::print("Think mode: {}\n", think_mode.isSet() ? "ENABLED" : "DISABLED");
    fmt::print("Debug mode: {}\n", debug_mode.isSet() ? "ENABLED" : "DISABLED");
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
        if (debug_mode.isSet()) {
          fmt::print("🔄 Processing...\n");
        }

        // 准备输入消息
        mllm::models::smollm3::SmolLM3Message message;
        message.prompt = prompt_text;
        message.enable_thinking = think_mode.isSet();

        // 转换消息为模型输入
        auto inputs = tokenizer.convertMessage(message);

        // 调试输出token信息
        if (debug_mode.isSet()) {
          auto sequence_tensor = inputs["sequence"];
          auto seq_ptr = sequence_tensor.ptr<int64_t>();
          
          fmt::print("=== DEBUG INFORMATION ===\n");
          fmt::print("Input Token IDs: ");
          for (int i = 0; i < sequence_tensor.shape()[1]; ++i) {
              fmt::print("{} ", seq_ptr[i]);
          }
          fmt::print("\n");
          fmt::print("Token count: {}\n", sequence_tensor.shape()[1]);
          fmt::print("Think mode: {}\n", message.enable_thinking);
          fmt::print("=======================\n\n");
        }

        fmt::print("🤖 SmolLM3: ");

        // 流式生成回复
        std::string full_response;
        int token_count = 0;
        bool in_thinking = think_mode.isSet();
        bool thinking_completed = false;
        bool response_started = false;
        
        for (auto& step : model.chat(inputs)) {
          std::wstring wtoken = tokenizer.detokenize(step.cur_token_id);
          std::string token_text(wtoken.begin(), wtoken.end());
          
          // Think模式处理 
          if (in_thinking && !thinking_completed) {
            // 检查是否遇到think结束标记
            if (token_text.find("</think>") != std::string::npos) {
              thinking_completed = true;
              in_thinking = false;
              fmt::print("\n🤖 Response: ");
              response_started = true;
              continue;
            }
            
            // 在think模式下显示思考内容
            if (think_mode.isSet()) {
              std::cout << token_text << std::flush;
            }
            continue;
          }
          
          // 正常输出回复
          if (!response_started && !think_mode.isSet()) {
            response_started = true;
          }
          
          std::cout << token_text << std::flush;
          full_response += token_text;
          token_count++;
          
          // 调试输出
          if (debug_mode.isSet()) {
            fmt::print("[{}] ", step.cur_token_id);
          }
          
          // 检查是否生成结束token
          if (step.cur_token_id == cfg.eos_token_id) {
            if (debug_mode.isSet()) {
              fmt::print("\n[EOS token detected]");
            }
            break;
          }
          
          // 安全限制
          if (token_count > 512) {
            if (debug_mode.isSet()) {
              fmt::print("\n[Token limit reached]");
            }
            fmt::print("\n[Output truncated]");
            break;
          }
        }

        fmt::print("\n{}\n", std::string(60, '-'));
        if (debug_mode.isSet()) {
          fmt::print("Generated {} tokens\n", token_count);
        }

        // 重置缓存
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
