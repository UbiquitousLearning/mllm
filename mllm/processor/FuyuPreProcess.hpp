//
// Created by Xiang Li on 2023/12/4.
//

#ifndef FUYUPREPROCESS_HPP
#define FUYUPREPROCESS_HPP
#include <utility>
#include <vector>

#include "PreProcess.hpp"
#include "tokenizers/Tokenizer.hpp"
// #include <stb/stb_image_resize.h>
using std::vector;

namespace mllm {
typedef vector<vector<vector<vector<float>>>> FourDVector;

class FuyuPreProcess : public PreProcessor {
    vector<vector<int>> image_patch_indices_per_batch;
    vector<vector<int>> image_patch_indices_per_subseq;
    vector<vector<int>> image_patch_input_indices_;
    vector<size_t> text_lengths_;
    size_t max_tokens_to_generate;
    token_id_t image_placeholder_id_;
    token_id_t image_newline_id_;
    std::pair<size_t, size_t> patch_size_;
    size_t max_position_embeddings = 16384;

    token_id_t pad_token_id = 0;
    int dummy_image_index = -1;

public:
    vector<vector<token_id_t>> image_input_ids_;
    vector<vector<int>> attention_mask_;
    vector<vector<int>> image_patches_indices_;
    vector<vector<vector<float>>> image_patches_;
    vector<vector<token_id_t>> text_ids_;
    std::vector<ImageInfo> images_;

    explicit FuyuPreProcess(Tokenizer *tokenizer, size_t patch_height = 30, size_t patch_width = 30, size_t max_tokens_to_generate = 10, int height = 1080, int width = 1920, bool do_pad = true, bool do_resize = true, bool do_normalize = true, std::vector<float> mean = {0.5}, std::vector<float> std = {0.5}) :
        PreProcessor(tokenizer, height, width, do_pad, do_resize, do_normalize, true, std::move(mean), std::move(std)), max_tokens_to_generate(max_tokens_to_generate) {
        auto tmp_token = vector<token_id_t>();
        tokenizer_->tokenize("|SPEAKER|", tmp_token, false);
        image_placeholder_id_ = tmp_token[0];
        tmp_token.clear();
        tokenizer_->tokenize("|NEWLINE|", tmp_token, false);
        image_newline_id_ = tmp_token[0];
        patch_size_ = std::make_pair(patch_height, patch_width);
    }

    void PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) override;
    void PreProcessImages(const std::vector<std::string> &images_path) override;
    void Process(const std::string &text) override;
    static std::vector<vector<float>> PatchImages(ImageInfo &images, size_t patch_height, size_t patch_width);

    // vector<Tensor> process(std::string &text, vector<string> image) override;
    // std::pair<std::string, unsigned> detokenize(Tensor &result) override;
    // std::pair<bool, std::string> postprocess(std::string &text) override;

private:
    void get_sample_encoding(const std::string &text);
    // vector<vector<token_id_t>> construct_full_unpacked_stream();
    void _left_pad_inputs_with_attention_mask();

    Tensor vector3d2Tensor(vector<vector<vector<float>>> image_patches, string name = "input", BackendType type = MLLM_CPU);
    Tensor vector2d2Tensor(vector<vector<int>> image_patches_indices, string name = "input", BackendType type = MLLM_CPU);
};
} // namespace mllm

#endif // FUYUPREPROCESS_HPP
