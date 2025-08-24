// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

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
MllmReturnCode mllm_init_context();

/**
 * @brief Shutdown the MLLM context
 * @return 0 on success, negative value on error
 */
MllmReturnCode mllm_shutdown_context();

/**
 * @brief Show memory report
 *
 * @return 0 on success, negative value on error
 */
MllmReturnCode mllm_show_memory_report();

enum ARGenerationStatusCode {
  AR_GENERATION_STATUS_CODE_SUCCESS = 0,
  AR_GENERATION_STATUS_CODE_EOF = 0,
};

struct ARGenerationContext {
  ARGenerationStatusCode status_code;
};

#ifdef __cplusplus
}
#endif
