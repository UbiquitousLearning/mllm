// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <memory.h>

#include <mllm/mllm.hpp>
#include <mllm/utils/Common.hpp>

// Qwen2
#include <mllm/models/qwen2vl/modeling_qwen2vl.hpp>
#include <mllm/models/qwen2vl/tokenization_qwen2vl.hpp>
#include <mllm/models/qwen2vl/configuration_qwen2vl.hpp>

#include "mllm-c.hpp"

namespace MLLM_ANONYMOUS_NAMESPACE {

void* loadARModel(const std::string& name, const std::string& model_file_path, char* config_file_path, int device) {
  if (name == "Qwen2VLForConditionalGeneration") {
    auto ret = new mllm::models::qwen2vl::Qwen2VLForCausalLM(mllm::models::qwen2vl::Qwen2VLConfig(config_file_path));
    ret->llm.to(mllm::DeviceTypes(device));
    ret->visual.to(mllm::DeviceTypes(device));
    return ret;
  }
  return nullptr;
}

void freeARModel(const std::string& name, void* model) {
  if (name == "Qwen2VLForConditionalGeneration") {
    delete static_cast<mllm::models::qwen2vl::Qwen2VLForCausalLM*>(model);
    return;
  }
}

void* loadARTokenizer(const std::string& name, const std::string& tokenizer_file_path) {
  if (name == "Qwen2VLForConditionalGeneration") {
    auto ret = new mllm::models::qwen2vl::Qwen2VLTokenizer(tokenizer_file_path);
    return ret;
  }
  return nullptr;
}

void freeARTokenizer(const std::string& name, void* tokenizer) {
  if (name == "Qwen2VLForConditionalGeneration") {
    delete static_cast<mllm::models::qwen2vl::Qwen2VLTokenizer*>(tokenizer);
    return;
  }
}

}  // namespace MLLM_ANONYMOUS_NAMESPACE

MllmReturnCode mllm_init_context() {
  mllm::initializeContext();
  return MLLM_SUCCESS;
}

MllmReturnCode mllm_shutdown_context() {
  mllm::shutdownContext();
  return MLLM_SUCCESS;
}

MllmReturnCode mllm_show_memory_report() {
  mllm::memoryReport();
  return MLLM_SUCCESS;
}

struct ARGenerationContext mllm_ar_from_pretrained(char* model_base_name, char* model_file_path, char* tokenizer_file_path,
                                                   char* config_file_path, int device) {
  ARGenerationContext ret;
  ret.model_handler = loadARModel(std::string(model_base_name), std::string(model_file_path), config_file_path, device);
  ret.tokenizer_handler = loadARTokenizer(std::string(model_base_name), std::string(tokenizer_file_path));
  return ret;
}

enum MllmReturnCode mllm_ar_context_free(struct ARGenerationContext* context) {
  if (context->model_handler) { freeARModel(std::string(context->model_file_name), context->model_handler); }
  if (context->tokenizer_handler) { freeARTokenizer(std::string(context->model_file_name), context->tokenizer_handler); }
  return MLLM_SUCCESS;
}

struct ARGenerationResult mllm_ar_step(struct ARGenerationContext* context) {
  ARGenerationResult ret;

  return ret;
}
