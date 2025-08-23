// Copyright (c) MLLM Team.
// Licensed under the MIT License.
#pragma once

#include <mllm/mllm.hpp>

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

#ifdef __cplusplus
}
#endif
