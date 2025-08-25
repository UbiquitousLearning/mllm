// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#define MLLM_MODEL_FILE_NAME_LEN 512

#ifdef __cplusplus
extern "C" {
#endif

enum MllmReturnCode {
  MLLM_SUCCESS = 0,
  MLLM_ERROR_INVALID_ARGUMENT = 1,
  MLLM_ERROR_OUT_OF_MEMORY = 2,
  MLLM_ERROR_UNKNOWN = 3,
};

/**
 * @brief Initialize the MLLM context
 * @return 0 on success, negative value on error
 */
enum MllmReturnCode mllm_init_context();

/**
 * @brief Shutdown the MLLM context
 * @return 0 on success, negative value on error
 */
enum MllmReturnCode mllm_shutdown_context();

/**
 * @brief Show memory report
 *
 * @return 0 on success, negative value on error
 */
enum MllmReturnCode mllm_show_memory_report();

enum ARGenerationStatusCode {
  AR_GENERATION_STATUS_CODE_SUCCESS = 0,
  AR_GENERATION_STATUS_CODE_EOF = 0,
};

struct ARGenerationContext {
  char model_file_name[MLLM_MODEL_FILE_NAME_LEN];
  enum ARGenerationStatusCode status_code;
  void* model_handler;
  void* tokenizer_handler;
};

struct ARGenerationResult {
  char* text;
};

struct ARGenerationContext mllm_ar_from_pretrained(char* model_base_name, char* model_file_path, char* tokenizer_file_path,
                                                   char* config_file_path, int device);

enum MllmReturnCode mllm_ar_context_free(struct ARGenerationContext* context);

struct ARGenerationResult mllm_ar_step(struct ARGenerationContext* context);

#ifdef __cplusplus
}
#endif
