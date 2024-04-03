/**
 * @file configuration_gemma.hpp
 * @author Chenghua Wang (chenghua.wang@gmail.com)
 * @brief configuration file of gemma llm.
 * @version 0.1
 * @date 2024-04-03
 *
 * @copyright Copyright (c) 2024
 *
 */

#ifndef CONFIG_GEMMA_HPP
#define CONFIG_GEMMA_HPP
#include "models/transformer/configuration_transformer.hpp"

using namespace mllm;

class GemmaNameConfig : public TransformerNameConfig {
public:
private:
};

struct GemmaConfig {
    explicit GemmaConfig(){

    };
};

#endif //! CONFIG_GEMMA_HPP
