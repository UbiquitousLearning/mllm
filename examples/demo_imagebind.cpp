//
// Created by Rongjie Yi on 24-2-29.
//
#include <iostream>
#include "cmdline.h"
#include "models/imagebind/modeling_imagebind.hpp"
#include "models/imagebind/processing_imagebind.hpp"

using namespace mllm;

int main(int argc, char **argv) {
    cmdline::parser cmdParser;
    cmdParser.add<string>("vocab", 'v', "specify mllm tokenizer model path", false, "../vocab/clip_vocab.mllm");
    cmdParser.add<string>("model", 'm', "specify mllm model path", false, "../models/imagebind_huge-q4_k.mllm");
    cmdParser.add<string>("merges", 'f', "specify mllm tokenizer merges.txt path", false, "../vocab/clip_merges.txt");
    cmdParser.add<int>("thread", 't', "num of threads", false, 4);
    cmdParser.parse_check(argc, argv);

    string vocab_path = cmdParser.get<string>("vocab");
    string model_path = cmdParser.get<string>("model");
    string merges_path = cmdParser.get<string>("merges");
    CPUBackend::cpu_threads = cmdParser.get<int>("thread");

    auto processor = ImagebindProcessor(vocab_path, merges_path);

    ImagebindConfig config("huge");
    auto model = ImagebindModel(config);
    model.load(model_path);

    auto input_tensors = processor.process(
        {"a dog.", "A car", "A bird"}, config.max_position_embeddings,
        {"../assets/dog_image.jpg", "../assets/car_image.jpg", "../assets/bird_image.jpg"}, config.img_hw,
        {"../assets/dog_audio.wav", "../assets/car_audio.wav", "../assets/bird_audio.wav"});
    auto result = model({input_tensors.text_tensors, input_tensors.img_tensors, input_tensors.audio_tensors}, input_tensors.in_len);
    processor.showResult(result);
}