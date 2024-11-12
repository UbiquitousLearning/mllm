//
// Created by Xiang Li on 2023/12/4.
//

#include "FuyuPreProcess.hpp"

#include <cassert>
#include <fstream>
#include <iostream>
#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#endif
#include "stb/stb_image.h"
#include "PreProcess.hpp"
#include "stb/stb_image_resize2.h"
namespace mllm {
void FuyuPreProcess::PreProcessImages(const std::vector<uint8_t *> &images, const std::vector<size_t> &image_length) {
    assert(height_ > 0 && width_ > 0);

    for (int i = 0; i < images.size(); i++) {
        auto image = images[i];
        int width_, height_, channels_;
        // Data is [height * width * channels],RGB
        const unsigned char *data = stbi_load_from_memory(image, image_length[i], &width_, &height_, &channels_, 3);
        if (data == nullptr) {
            MLLM_LOG_ERROR_STREAM << "load image failed" << std::endl;
            exit(-1);
        }
        auto float_data = RescaleImage(data, 255.0, width_ * height_ * channels_);
        images_.emplace_back(float_data, width_, height_, channels_);
    }
    auto image_patches = std::vector<FourDVector>();
    if (do_resize_) {
        images_ = ResizeImages(images_, height_, width_);
    }

    // TODO: PAD images
    if (do_pad_) {
        images_ = PadImages(images_, height_, width_, patch_size_.second, patch_size_.first);
    }
    if (do_normalize_) {
        if (mean_.size() != std_.size() || mean_.size() != 1 && mean_.size() != 3) {
            MLLM_LOG_ERROR_STREAM << "MEAN should be of same size of std and length should be (1 or 3) !" << std::endl;
            exit(-1);
        }
        if (mean_.size() == 1) {
            mean_.resize(3, mean_[0]);
        }
        if (std_.size() == 1) {
            std_.resize(3, std_[0]);
        }
        images_ = NormalizeImages(images_, mean_, std_);
    }
}

void FuyuPreProcess::PreProcessImages(const std::vector<std::string> &images_path) {
    assert(height_ > 0 && width_ > 0);
    auto image_data = std::vector<uint8_t *>();
    auto image_length = std::vector<size_t>();
    for (const auto &i : images_path) {
        // read all file contents
        std::ifstream file(i, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            MLLM_LOG_ERROR_STREAM << "Cannot open file: " << i << std::endl;
            exit(-1);
        }
        auto size = file.tellg();
        auto data = new uint8_t[size];
        file.seekg(0, std::ios::beg);
        file.read(reinterpret_cast<char *>(data), size);
        file.close();
        image_data.emplace_back(data);
        image_length.emplace_back(size);
    }
    PreProcessImages(image_data, image_length);
}

void FuyuPreProcess::Process(const std::string &text) {
    if (text.empty()) {
        return;
    }
    if (images_.empty()) {
        std::cout << "images is empty" << std::endl;
    }
    // auto batch_size = images_.size();
    get_sample_encoding(text);
}

void FuyuPreProcess::get_sample_encoding(const std::string &text) {
    image_input_ids_.resize(images_.size());
    image_patches_.resize(images_.size());
    image_patch_indices_per_batch.resize(images_.size());
    image_patch_indices_per_subseq.resize(images_.size());
    auto num_index = 0;
    for (int i = 0; i < images_.size(); i++) {
        auto image = images_[i];
        // for (auto &image : images_) {
        auto height = image.height;
        auto width = image.width;
        auto num_patches_per_dim_h = height / patch_size_.first;
        auto num_patches_per_dim_w = width / patch_size_.second;
        auto num_patches = num_patches_per_dim_h * num_patches_per_dim_w;
        auto tensor_of_image_ids = vector<vector<token_id_t>>(num_patches_per_dim_h, vector<token_id_t>(num_patches_per_dim_w, image_placeholder_id_));
        auto &image_input_id = image_input_ids_[i];
        auto &image_patch_indices_per_batch_ = image_patch_indices_per_batch[i];
        auto &image_patch_indices_per_subseq_ = image_patch_indices_per_subseq[i];
        for (int h = 0; h < num_patches_per_dim_h; h++) {
            for (int w = 0; w < num_patches_per_dim_w; w++) {
                auto patch_index = h * num_patches_per_dim_w + w;
                image_patch_indices_per_batch_.push_back(patch_index + num_index);
                image_patch_indices_per_subseq_.push_back(patch_index);
            }
            image_patch_indices_per_batch_.emplace_back(-1);
            image_patch_indices_per_subseq_.emplace_back(-1);
        }
        num_index += num_patches;

        for (auto &row : tensor_of_image_ids) {
            row.push_back(image_newline_id_);
            image_input_id.insert(image_input_id.end(), row.begin(), row.end());
        }
        // image_input_id.pop_back();
        image_patches_[i] = PatchImages(image, patch_size_.first, patch_size_.second);
    }
    // TODO: _transform_coordinates_and_tokenize
    //_tokenize_prompts_with_image_and_batch
    // Now handle the text
    // TODO: More than One line of text.
    // tokenizer_->setSpecialToken("|ENDOFTEXT|");
    tokenizer_->setSpecialToken("<s>");

    auto text_ = Tokenizer::replaceString(text, ' ', "▁");
    text_ = "▁" + text_;
    auto text_ids = vector<token_id_t>();
    // token_id_t bos_token_id = 0;
    // if (tokenizer_->getTokenId("<s>", bos_token_id)) {
    //     text_ids.push_back(bos_token_id);
    // } else {
    //     MLLM_LOG_ERROR_STREAM << "BOS token not found in vocab file." << std::endl;
    // }
    tokenizer_->tokenize(text_, text_ids, true);
    token_id_t answer_start_token = 0;
    if (tokenizer_->getTokenId("<0x04>", answer_start_token)) {
        text_ids.push_back(answer_start_token);
    } else {
        MLLM_LOG_ERROR_STREAM << "ANSWER_START token not found in vocab file." << std::endl;
    }
    // token_id_t end_of_text_token = 0;
    // if (tokenizer_->getTokenId("|ENDOFTEXT|", end_of_text_token)) {
    //     text_ids.push_back(end_of_text_token);
    // } else {
    //     MLLM_LOG_ERROR_STREAM << "END_OF_TEXT token not found in vocab file." << std::endl;
    // }
    text_ids_.push_back(text_ids);
    // TODO: Should we Pad the prompt tokens? HF pad & cut off the padding in `construct_full_unpacked_stream`.
    text_lengths_.push_back(text_ids.size());
    // construct_full_unpacked_stream
    auto image_padded_unpacked_tokens = vector<vector<token_id_t>>(images_.size(), vector<token_id_t>());
    auto unpacked_image_patch_indices_per_batch = vector<vector<int>>(images_.size(), vector<int>());
    size_t max_prompt_length = 0;
    for (int i = 0; i < images_.size(); i++) {
        auto &image_padded_unpacked_token = image_padded_unpacked_tokens[i];
        auto &unpacked_image_patch_indice_per_batch = unpacked_image_patch_indices_per_batch[i];
        // TODO:
        auto text_length = text_lengths_[0];
        auto image_token_length = image_input_ids_[i].size();
        if (text_lengths_.size() > 1) {
            text_length = text_lengths_[i];
        }
        auto size_ = image_token_length + text_length;
        image_padded_unpacked_token.insert(image_padded_unpacked_token.begin(), image_input_ids_[i].begin(), image_input_ids_[i].end());
        image_padded_unpacked_token.insert(image_padded_unpacked_token.end(), text_ids_[0].begin(), text_ids_[0].end());
        unpacked_image_patch_indice_per_batch.insert(unpacked_image_patch_indice_per_batch.begin(), image_patch_indices_per_batch[i].begin(), image_patch_indices_per_batch[i].end());
        unpacked_image_patch_indice_per_batch.insert(unpacked_image_patch_indice_per_batch.end(), text_ids_[0].size(), -1);
        if (size_ > max_prompt_length) {
            max_prompt_length = size_;
        }
        //
    }
    size_t max_seq_len_batch = std::min(max_prompt_length + max_tokens_to_generate, max_position_embeddings);
    auto tokens_to_place = std::min(max_seq_len_batch, max_prompt_length);
    // full_unpacked_stream_to_tensor
    image_patch_input_indices_.resize(images_.size());
    // for (auto &image_patch_input_indice : image_patch_input_indices_) {
    for (int i = 0; i < image_patch_input_indices_.size(); i++) {
        auto &image_patch_input_indice = image_patch_input_indices_[i];
        image_patch_input_indice.insert(image_patch_input_indice.begin(), unpacked_image_patch_indices_per_batch[i].begin(), unpacked_image_patch_indices_per_batch[i].begin() + tokens_to_place);
        image_patch_input_indice.insert(image_patch_input_indice.end(), max_seq_len_batch - tokens_to_place, -1);
    }
    image_input_ids_.clear();
    image_input_ids_.resize(images_.size());
    //_left_pad_inputs_with_attention_mask
    attention_mask_.resize(images_.size());
    image_patches_indices_.resize(images_.size());
    for (int i = 0; i < images_.size(); i++) {
        auto &attention_mask = attention_mask_[i];
        auto &input_id = image_input_ids_[i];
        auto num_padding_tokens = max_prompt_length - image_padded_unpacked_tokens[i].size();
        input_id.insert(input_id.end(), num_padding_tokens, pad_token_id);
        input_id.insert(input_id.end(), image_padded_unpacked_tokens[i].begin(), image_padded_unpacked_tokens[i].end());
        attention_mask.insert(attention_mask.end(), num_padding_tokens, 0);
        attention_mask.insert(attention_mask.end(), image_padded_unpacked_tokens[i].size(), 1);

        // For the image patches indices, we need to add the padding tokens as well.
        auto &image_patch_input_indice = image_patches_indices_[i];
        auto &image_patch_input_indice_per_batch = image_patch_input_indices_[i];
        auto num_padding_indices = max_seq_len_batch - image_patch_input_indice_per_batch.size();
        image_patch_input_indice.insert(image_patch_input_indice.end(), num_padding_indices, -1);
        image_patch_input_indice.insert(image_patch_input_indice.end(), image_patch_input_indice_per_batch.begin(), image_patch_input_indice_per_batch.end());
    }
}

std::vector<vector<float>> FuyuPreProcess::PatchImages(ImageInfo &images, size_t patch_height, size_t patch_width) {
    // auto batch = images.size();
    // if (batch == 0) {
    //     return {};
    // }
    auto image_0 = images;
    auto height = image_0.height;
    auto width = image_0.width;
    auto channels = image_0.channels;
    auto square = width * height;
    auto dim2 = square / patch_height / patch_width;
    auto dim_2_1 = width / patch_width;
    auto dim2_2 = height / patch_height;
    auto stride2 = patch_height * width;
    auto stride1 = patch_width;
    auto patches = vector<vector<float>>(dim2, vector<float>());
    // for (int b = 0; b < batch; b++) {
    for (int i = 0; i < dim2_2; i++) {
        for (int j = 0; j < dim_2_1; j++) {
            auto &patch = patches[i * dim_2_1 + j];
            auto const index_first_element_of_line = i * stride2 + j * stride1;
            while (patch.size() < patch_height * patch_width * channels) {
                for (int h = 0; h < patch_height; h++) {
                    for (int w = 0; w < patch_width; w++) {
                        for (int c = 0; c < channels; c++) {
                            patch.push_back(images.get_whc_pixel(index_first_element_of_line + h * width + w + c * square));
                        }
                    }
                }
            }
        }
    }
    // if (free_source) {
    //     free(images[b].data);
    //     images[b].data = nullptr;
    // }

    return patches;
}

void FuyuPreProcess::_left_pad_inputs_with_attention_mask() {
}

Tensor FuyuPreProcess::vector3d2Tensor(vector<vector<vector<float>>> image_patches, string name, BackendType type) {
    int batch = 0;
    int seq = 0;
    int dims = 0;
    if (!image_patches.empty()) {
        batch = image_patches.size();
        seq = image_patches[0].size();
        dims = image_patches[0][0].size();
    }
    Tensor tensor1(batch, 1, seq, dims, Backend::global_backends[type], true);
    tensor1.setName(name);
    Tensor::tensor_status = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq; ++j) {
            for (int k = 0; k < dims; ++k) {
                tensor1.setDataAt<float>(i, 0, j, k, image_patches[i][j][k]);
            }
        }
    }
    return tensor1;
}

Tensor FuyuPreProcess::vector2d2Tensor(vector<vector<int>> image_patches_indices, string name, BackendType type) {
    int batch = 0;
    int seq = 0;
    if (!image_patches_indices.empty()) {
        batch = image_patches_indices.size();
        seq = image_patches_indices[0].size();
    }
    Tensor tensor1(batch, 1, seq, 1, Backend::global_backends[type], true);
    tensor1.setName(name);
    Tensor::tensor_status = TENSOR_STATIC_INIT;
    tensor1.setTtype(INPUT_TENSOR);
    for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < seq; ++j) {
            tensor1.setDataAt<float>(i, 0, j, 0, image_patches_indices[i][j]);
        }
    }
    return tensor1;
}

// vector<Tensor> FuyuPreProcess::process(std::string &text, vector<string> image) {
//     this->images_.clear();
//     this->image_input_ids_.clear();
//     this->image_patches_indices_.clear();
//     this->image_patches_.clear();
//     this->image_patch_input_indices_.clear();
//     this->PreProcessImages(image);
//     this->Process(text);
//     auto input_ids = this->image_input_ids_;
//     auto image_patches_indices = this->image_patches_indices_;
//     auto image_patches = this->image_patches_;
//     if (input_ids.empty()) {
//         input_ids = this->text_ids_;
//     }
//     vector<Tensor> result = {mllm::Tokenizer::tokens2Input(input_ids[0], "input_ids"),
//                              this->vector3d2Tensor(image_patches, "image_patches"),
//                              this->vector2d2Tensor(image_patches_indices, "image_patches_indices")};
//     return result;
// }
// std::pair<std::string, unsigned> FuyuPreProcess::detokenize(Tensor &result) {
//     return tokenizer_->detokenize(result);
// }

// std::pair<bool, std::string> FuyuPreProcess::postprocess(std::string &text) {
//     size_t pos = 0;
//     std::string from = "▁";
//     std::string to = " ";
//     while ((pos = text.find(from, pos)) != std::string::npos) {
//         text.replace(pos, from.length(), to);
//         pos += to.length();
//     }
//     if (text == "|ENDOFTEXT|") return {false, ""};
//     return {true, text};
// }

} // namespace mllm
