/**
 * @file modeling_gemma.hpp
 * @author Chenghua Wang (chenghua.wang@gmail.com)
 * @brief The defination of gemma model
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef MODELING_GEMMA_HPP
#define MODELING_GEMMA_HPP

#include "Layer.hpp"
#include "Module.hpp"
#include "configuration_gemma.hpp"
#include "models/transformer/modeling_transformer.hpp"

using namespace mllm;

class GemmaMLP final : public Module {};

class GemmaAttention final : public Module {};

class GemmaDecoder final : public Module {};

class GemmaModle final : public Module {};

#endif //! MODELING_GEMMA_HPP