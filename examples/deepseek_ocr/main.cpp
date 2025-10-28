#include <mllm/mllm.hpp>
#include "mllm/models/deepseek_ocr/modeling_deepseek_ocr.hpp"
#include "mllm/models/deepseek_ocr/tokenization_deepseek_ocr.hpp"

using mllm::Argparse;

MLLM_MAIN({
  // auto config = mllm::models::deepseek_ocr::DpskOcrConfig("/Volumes/D/mllm-models/DeepSeek-OCR-w32a32/config.json");
  // auto model = mllm::models::deepseek_ocr::DeepseekOCRForCausalLM(config);
  // auto tokenizer = mllm::models::deepseek_ocr::DpskOcrTokenizer("/Volumes/D/mllm-models/DeepSeek-OCR-w32a32/tokenizer.json");
  // model.load(mllm::load("/Volumes/D/mllm-models/DeepSeek-OCR-w32a32/model.mllm", mllm::ModelFileVersion::kV2));
  mllm::setLogLevel(mllm::LogLevel::kError);
  auto config = mllm::models::deepseek_ocr::DpskOcrConfig("/Volumes/D/mllm-models/DeepSeek-OCR-w4a8-i8mm-kai/config.json");
  auto model = mllm::models::deepseek_ocr::DeepseekOCRForCausalLM(config);
  auto tokenizer =
      mllm::models::deepseek_ocr::DpskOcrTokenizer("/Volumes/D/mllm-models/DeepSeek-OCR-w4a8-i8mm-kai/tokenizer.json");
  model.load(mllm::load("/Volumes/D/mllm-models/DeepSeek-OCR-w4a8-i8mm-kai/model.mllm", mllm::ModelFileVersion::kV2));

  model.infer(tokenizer, "<image>\n<|grounding|>Convert the document to markdown. ", "/Volumes/D/mllm/.tmp/dpsk-ocr-pr.png",
              "/Volumes/D/mllm/.tmp/dpsk-ocr");
});
