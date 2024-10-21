//
// Created by Xiang Li on 2023/12/29.
//

#ifndef CLIPPREPROCESS_HPP
#define CLIPPREPROCESS_HPP
#include "PreProcess.hpp"
namespace mllm {
class ClipPreProcessor : public mllm::PreProcessor {
public:
    explicit ClipPreProcessor(
        mllm::Tokenizer *tokenizer, int height = 224, int width = 224,
        bool do_pad = false, bool do_resize = true, bool do_normalize = true,
        bool do_rescale = true,
        std::vector<float> mean = {0.48145466, 0.4578275, 0.40821073},
        std::vector<float> std = {0.26862954, 0.26130258, 0.27577711}) :
        PreProcessor(tokenizer, height, width, do_pad, do_resize, do_normalize,
                     do_rescale, std::move(mean), std::move(std)) {
    }
    vector<vector<token_id_t>> input_ids_;
    vector<vector<int>> attention_mask_;
    // 4-D vector
    vector<vector<vector<vector<float>>>> pixel_values_;
    void Process(const std::string &text) override;
    void PreProcessImages(const std::vector<uint8_t *> &images,
                          const std::vector<size_t> &image_length) override;
    void PreProcessImages(const std::vector<std::string> &images_path) override;
    void PreProcessImages(const std::vector<std::string> &images_path, int height, int width) {
        height_ = height;
        width_ = width;
        PreProcessImages(images_path);
    }

    void Img2Tensor(Backend *bn, shared_ptr<Tensor> input_tensor, vector<vector<vector<float>>> img);

    // vector<Tensor> process(std::string &text, vector<string> image) override;
    // std::pair<std::string, unsigned> detokenize(Tensor &result) override;
    // std::pair<bool, std::string> postprocess(std::string &text) override;
};
} // namespace mllm
#endif // CLIPPREPROCESS_HPP
