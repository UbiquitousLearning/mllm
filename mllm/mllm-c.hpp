// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the MLLM context
 * @return 0 on success, negative value on error
 */
int mllm_init_context();

/**
 * @brief Shutdown the MLLM context
 * @return 0 on success, negative value on error
 */
int mllm_shutdown_context();

/**
 * @brief Show memory report
 *
 * @return int
 */
int mllm_show_memory_report();

#ifdef __cplusplus
}
#endif
