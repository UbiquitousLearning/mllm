#include <iostream>
#include <fmt/core.h>
#include <mllm/mllm.hpp>
#include "mllm/models/deepseek_ocr/modeling_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto model = mllm::models::deepseek_ocr::DeepseekOCRForCausalLM();
  auto tokenizer = mllm::models::deepseek_ocr::DpskOcrTokenizer("/Volumes/D/hf-models/DeepSeek-OCR/tokenizer.json");
  model.infer(tokenizer, "<image>\n<|grounding|>Convert the document to markdown. ", "/Volumes/D/mllm/.tmp/dpsk-ocr-pr.png",
              "/Volumes/D/mllm/.tmp/dpsk-ocr");
});
