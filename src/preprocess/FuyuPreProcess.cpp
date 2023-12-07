//
// Created by 咸的鱼 on 2023/12/4.
//

#include "FuyuPreProcess.hpp"

#include <iostream>

namespace mllm {
void FuyuPreProcess::PreProcessImages(const std::vector<vector<uint8_t>> &images, int height, int width, bool do_pad, bool do_resize, bool do_normalize, float mean, float std) {
    assert(height > 0 && width > 0);

    // if (do_resize) {
    //     // Not implemented yet
    //     std::cerr << "Resize not implemented yet" << std::endl;
    //     exit(-1);
    // }
    // auto images_ = std::vector<ImageInfo>();
    images_.resize(images.size());
    for (auto image : images) {
        int width_, height_, channels_;
        // Data is [height * width * channels],RGB
        const unsigned char *data = stbi_load_from_memory(image.data(), image.size(), &width, &height, &channels_, 3);
        if (data == nullptr) {
            std::cerr << "load image failed" << std::endl;
            exit(-1);
        }
        float *float_data = new float[width * height * channels_];
        for (int i = 0; i < width * height * channels_; i++) {
            float_data[i] = data[i] / 255.0;
        }

        images_.emplace_back(float_data, width_, height_, channels_);
    }
    auto image_patches = std::vector<FourDVector>();
    if (do_resize) {
        images_ = ResizeImages(images_, height, width);
    }

    // TODO: PAD images
    if (do_pad) {
        images_ = PadImages(images_, height, width);
    }
    if (do_normalize) {
        images_ = NormalizeImages(images_, mean, std);
    }
    // TODO: Patch images
}

void FuyuPreProcess::Process(std::string text) {
    if (text.empty()) {
        return;
    }
    if (images_.empty()) {
        std::cout << "images is empty" << std::endl;
    }
    auto batch_size = images_.size();
}

std::vector<ImageInfo> FuyuPreProcess::PadImages(const std::vector<ImageInfo> &images, int height, int width, float pad_value, PaddingType padding_type, bool free_source) {
    assert(padding_type == PaddingType::CONSTANT);
    auto padded_images = std::vector<ImageInfo>();
    for (auto image : images) {
        if (image.height == height && image.width == width && image.channels == 3) {
            padded_images.emplace_back(image.data, image.width, image.height, image.channels, image.original_width, image.original_height);

            continue;
        }
        auto padded_image = std::vector<float>();
        padded_image.resize(height * width * image.channels, pad_value);
        for (int i = 0; i < image.height; i++) {
            for (int j = 0; j < image.width; j++) {
                for (int k = 0; k < image.channels; k++) {
                    padded_image[(i * width + j) * image.channels + k] = image.data[(i * image.width + j) * image.channels + k];
                }
            }
        }
        padded_images.emplace_back(padded_image.data(), width, height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        delete &images;
    }
    return padded_images;
}

std::vector<ImageInfo> FuyuPreProcess::ResizeImages(const std::vector<ImageInfo> &images, int height, int width, ResampleType resample_type, bool free_source) {
    assert(resample_type == ResampleType::BILINEAR);
    assert(free_source=true);
    stbir_filter filter = stbir_filter::STBIR_FILTER_DEFAULT;
    switch (resample_type) {
    case ResampleType::BILINEAR: filter = stbir_filter::STBIR_FILTER_TRIANGLE;
        break;
    default: std::cerr << "Not implemented Reshape Filter yet" << std::endl;
    // exit(-1);
        filter = stbir_filter::STBIR_FILTER_DEFAULT;
    }
    auto resized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        auto resized_image = std::vector<float>();
        resized_image.resize(height * width * image.channels);
        stbir_resize(image.data, image.width, image.height, 0, resized_image.data(), width, height, 0, stbir_pixel_layout::STBIR_RGB, stbir_datatype::STBIR_TYPE_FLOAT, stbir_edge::STBIR_EDGE_CLAMP, filter);
        resized_images.emplace_back(resized_image.data(), width, height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        delete &images;
    }

    return resized_images;
}

vector<ImageInfo> FuyuPreProcess::NormalizeImages(const vector<ImageInfo> &images, float mean, float std, bool free_source) {
    auto normalized_images = std::vector<ImageInfo>();
    for (auto image : images) {
        auto normalized_image = std::vector<float>();
        normalized_image.resize(image.width * image.height * image.channels);
        for (int i = 0; i < image.width * image.height * image.channels; i++) {
            normalized_image[i] = (image.data[i] - mean) / std;
        }
        normalized_images.emplace_back(normalized_image.data(), image.width, image.height, image.channels, image.original_width, image.original_height);
        if (free_source) {
            free(image.data);
            image.data = nullptr;
        }
    }
    if (free_source) {
        delete &images;
    }
    return normalized_images;
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
    //Now handle the text
    //TODO: More than One line of text.
    tokenizer_->setSpecialToken("|ENDOFTEXT|");
    auto text_ = Tokenizer::replaceString(text, ' ', "▁");
    text_ = "▁" + text_;
    auto text_ids = vector<token_id_t>();
    token_id_t bos_token_id = 0;
    if (tokenizer_->getTokenId("<s>", bos_token_id)) {
        text_ids.push_back(bos_token_id);
    } else {
        std::cerr << "BOS token not found in vocab file." << std::endl;
    }
    tokenizer_->tokenize(text_, text_ids, true);
    token_id_t answer_start_token = 0;
    if (tokenizer_->getTokenId("token_id_t", answer_start_token)) {
        text_ids.push_back(answer_start_token);
    } else {
        std::cerr << "ANSWER_START token not found in vocab file." << std::endl;
    }
    token_id_t end_of_text_token = 0;
    if (tokenizer_->getTokenId("|ENDOFTEXT|", end_of_text_token)) {
        text_ids.push_back(end_of_text_token);
    } else {
        std::cerr << "END_OF_TEXT token not found in vocab file." << std::endl;
    }
    text_ids_.push_back(text_ids);
    //TODO: Should we Pad the prompt tokens? HF pad & cut off the padding in `construct_full_unpacked_stream`.
    text_lengths_.push_back(text_ids.size());
//construct_full_unpacked_stream
    auto image_padded_unpacked_tokens = vector<vector<token_id_t>>(images_.size(), vector<token_id_t>());
    auto unpacked_image_patch_indices_per_batch = vector<vector<int>>(images_.size(), vector<int>());
    size_t max_prompt_length = 0;
    for (int i = 0; i < images_.size(); i++) {
        auto &image_padded_unpacked_token = image_padded_unpacked_tokens[i];
        auto &unpacked_image_patch_indice_per_batch = unpacked_image_patch_indices_per_batch[i];
        //TODO:
        auto text_length = text_lengths_[0];
        auto image_token_length = image_input_ids_[i].size();
        if (text_lengths_.size()>1) {
            text_length = text_lengths_[i];
        }
        auto size_ =image_token_length+text_length;
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
    //full_unpacked_stream_to_tensor
    image_patch_input_indices_.resize(images_.size());
    // for (auto &image_patch_input_indice : image_patch_input_indices_) {
    for (int i = 0; i < image_patch_input_indices_.size(); i++) {
        auto &image_patch_input_indice = image_patch_input_indices_[i];
        image_patch_input_indice.insert(image_patch_input_indice.begin(), unpacked_image_patch_indices_per_batch[i].begin(), unpacked_image_patch_indices_per_batch[i].begin()+tokens_to_place);
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
       auto  num_padding_tokens = max_prompt_length - image_padded_unpacked_tokens[i].size();
        input_id.insert(input_id.end(), num_padding_tokens, pad_token_id);
        input_id.insert(input_id.end, image_padded_unpacked_tokens[i].begin(), image_padded_unpacked_tokens[i].end());
        attention_mask.insert(attention_mask.end(), num_padding_tokens, 0);
        attention_mask.insert(attention_mask.end(), image_padded_unpacked_tokens[i].size(), 1);

        // For the image patches indices, we need to add the padding tokens as well.
        auto &image_patch_input_indice = image_patches_indices_[i];
        auto &image_patch_input_indice_per_batch = image_patch_input_indices_[i];
        auto num_padding_indices  = max_seq_len_batch - image_patch_input_indice_per_batch.size();
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
            auto patch = patches[i * dim_2_1 + j];
            auto const index_first_element_of_line = i * stride2 + j * stride1;
            while (patch.size() < patch_height * patch_width * channels) {
                for (int h = 0; h < patch_height; h++) {
                    for (int w = 0; w < patch_width; w++) {
                        for (int c = 0; c < channels; c++) {
                            patch.push_back(images.data[index_first_element_of_line + h * patch_height + w + c * square]);
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
} // mllm
