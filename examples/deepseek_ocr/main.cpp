#include <mllm/mllm.hpp>
#include "mllm/models/deepseek_ocr/modeling_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"

using mllm::Argparse;

MLLM_MAIN({
  auto model = mllm::models::deepseek_ocr::DeepseekOCRForCausalLM();
  auto tokenizer = mllm::models::deepseek_ocr::DpskOcrTokenizer("/Volumes/D/hf-models/DeepSeek-OCR/tokenizer.json");

  mllm::print(tokenizer.tokenize("<image>\n<|grounding|>Convert the document to markdown. "));
  mllm::print(tokenizer.encode("<image>\n<|grounding|>Convert the document to markdown. "));
  mllm::print(tokenizer.decode({128815, 201, 128820, 21842, 270, 4940, 304, 2121, 7919, 16, 223}));
  exit(0);

  model.infer(tokenizer, "<image>\n<|grounding|>Convert the document to markdown. ", "/Volumes/D/mllm/.tmp/dpsk-ocr-pr.png",
              "/Volumes/D/mllm/.tmp/dpsk-ocr");
});
