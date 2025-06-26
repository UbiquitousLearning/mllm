//
// Created by Rongjie Yi on 24-2-29.
//

#ifndef PROCESSING_IMAGEBIND_HPP
#define PROCESSING_IMAGEBIND_HPP
#include <utility>

#include "tokenizers/BPE/Bpe.hpp"
#include "models/clip/processing_clip.hpp"

using namespace mllm;

class ImagebindProcessor final : public ClipProcessor {
    static Tensor tokens2Input(vector<vector<token_id_t>> tokens, int max_pos, string name = "input", BackendType type = MLLM_CPU) {
        const auto bsize = static_cast<int>(tokens.size());
        Tensor tensor1(bsize, 1, max_pos, 1, Backend::global_backends[type].get(), true);
        tensor1.setName(name);
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int b = 0; b < bsize; ++b) {
            for (int idx = 0; idx < max_pos; ++idx) {
                if (idx < tokens[b].size()) {
                    tensor1.setDataAt<float>(b, 0, idx, 0, tokens[b][idx]);
                } else {
                    tensor1.setDataAt<float>(b, 0, idx, 0, 0);
                }
            }
        }
        return tensor1;
    }
    static Tensor img2Tensor(vector<vector<vector<vector<float>>>> imgs, string name = "input", BackendType type = MLLM_CPU) {
        int channel = imgs[0].size();
        int height = imgs[0][0].size();
        int width = imgs[0][0][0].size();
        Tensor tensor1(Backend::global_backends[type].get());
        tensor1.reshape(imgs.size(), channel, 2, height, width);
        tensor1.setDtype(MLLM_TYPE_F32);
        tensor1.alloc();
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);
        for (int bi = 0; bi < imgs.size(); ++bi) {
            for (int t = 0; t < 2; ++t) {
                for (int h = 0; h < height; ++h) {
                    for (int c = 0; c < channel; ++c) {
                        for (int w = 0; w < width; ++w) {
                            tensor1.setDataAt<float>(bi, c, t, h, w, imgs[bi][c][h][w]);
                        }
                    }
                }
            }
        }
        return tensor1;
    }
    static Tensor audio2Tensor(vector<vector<vector<vector<float>>>> audio, string name = "input", BackendType type = MLLM_CPU) {
        vector<vector<vector<float>>> audio_new;
        for (auto auv : audio) {
            for (auto au : auv) {
                audio_new.push_back(au);
            }
        }
        int batch = audio_new.size();
        int channel = 1;
        int height = audio_new[0].size();
        int width = audio_new[0][0].size();

        Tensor tensor1(batch, height, channel, width, Backend::global_backends[type].get(), true);
        tensor1.setName(std::move(name));
        Tensor::tensor_status = TENSOR_STATIC_INIT;
        tensor1.setTtype(INPUT_TENSOR);

        for (int bi = 0; bi < audio_new.size(); ++bi) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    tensor1.setDataAt<float>(bi, h, 0, w, audio_new[bi][h][w]);
                }
            }
        }
        return tensor1;
    }

    std::string toLowercase(const std::string &input) {
        std::string output = input;
        std::transform(output.begin(), output.end(), output.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return output;
    }

public:
    explicit ImagebindProcessor(const string &vocab_path, const string &merges_path) :
        ClipProcessor(vocab_path, merges_path) {
        Module::initBackend(MLLM_CPU);
    }

    struct imagebind_out {
        Tensor text_tensors;
        Tensor img_tensors;
        Tensor audio_tensors;
        vector<int> in_len;
    };
    imagebind_out process(vector<string> in_strs, int max_pos, vector<string> img_path, int hw, vector<string> wav_path,
                          string text_name = "input_text", string img_name = "input_vision", string wav_name = "input_audio",
                          BackendType type = MLLM_CPU) {
        auto tokens_ids = vector<vector<token_id_t>>();
        for (auto in_str : in_strs) {
            in_str = toLowercase(in_str);
            vector<mllm::token_id_t> tokens_id = {};
            tokenizer->tokenize(in_str, tokens_id, true, true, "</w>");
            tokens_ids.push_back(tokens_id);
        }
        vector<int> input_text_lens = {};
        for (auto tokens_id : tokens_ids) {
            input_text_lens.push_back(tokens_id.size() - 1);
        }

        PreProcessImages(img_path, hw, hw);
        auto images = pixel_values_;

        auto audios = PreProcessor::ProcessAudio(std::move(wav_path));

        return {tokens2Input(tokens_ids, max_pos, std::move(text_name)),
                img2Tensor(images, std::move(img_name)),
                audio2Tensor(audios, std::move(wav_name)), input_text_lens};
    }

    void showResult(Tensor &tensor) {
        // std::cout<<"vision X text :"<<std::endl;
        // std::cout<<std::endl;
        for (int s = 0; s < tensor.sequence(); ++s) {
            for (int d = 0; d < tensor.dimension(); ++d) {
                std::cout << tensor.dataAt<float>(0, 0, s, d) << " ";
            }
            std::cout << std::endl;
        }
        // std::cout<<"vision X audio :"<<std::endl;
        // for (int s = 0; s < tensor.sequence(); ++s) {
        //     for (int d = 0; d < tensor.dimension(); ++d) {
        //         std::cout<<tensor.dataAt<float>(1, 0, s, d)<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
    }
    void showResult(vector<Tensor> tensors) {
        vector<string> shows = {"vision X text :", "vision X audio :"};
        for (int i = 0; i < tensors.size(); ++i) {
            std::cout << shows[i] << std::endl;
            showResult(tensors[i]);
        }
    }
};

#endif // PROCESSING_IMAGEBIND_HPP
