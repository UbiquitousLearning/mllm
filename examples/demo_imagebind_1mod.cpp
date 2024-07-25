//
// Created by Rongjie Yi on 24-7-15.
//
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

    int loop_times = 10;

    // auto input_tensors = processor.process(
    //     {"a dog.", "A car", "A bird"},config.max_position_embeddings,
    //     {"../assets/dog_image.jpg", "../assets/car_image.jpg", "../assets/bird_image.jpg"}, config.img_hw,
    //     {"../assets/dog_audio.wav", "../assets/car_audio.wav", "../assets/bird_audio.wav"});
    
    auto input_tensors = processor.process(
        {"a dog."},config.max_position_embeddings,
        {"../assets/dog_image.jpg"}, config.img_hw,
        {"../assets/dog_audio.wav"});
    
    std::cout<<"Text| input_shape:["<<input_tensors.text_tensors.batch()<<", "<<input_tensors.text_tensors.sequence()<<", "<<input_tensors.text_tensors.head()<<", "<<input_tensors.text_tensors.dimension()<<"]"<<std::endl;
    auto text_model = ImagebindTextModel(config);
    text_model.load(model_path);
    for (int step = 0; step < loop_times; step++) {
        auto result = text_model({input_tensors.text_tensors}, input_tensors.in_len);
    }
    text_model.profiling();

    std::cout<<"Vision| input_shape:["<<input_tensors.img_tensors.batch()<<", "<<input_tensors.img_tensors.channel()<<", "<<input_tensors.img_tensors.time()<<", "<<input_tensors.img_tensors.height()<<", "<<input_tensors.img_tensors.width()<<"]"<<std::endl;
    auto vision_model = ImagebindVisionModel(config);
    vision_model.load(model_path);    
    for (int step = 0; step < loop_times; step++) {
        auto result = vision_model({input_tensors.img_tensors});
    }
    vision_model.profiling();

    std::cout<<"Audio| input_shape:["<<input_tensors.audio_tensors.batch()<<", "<<input_tensors.audio_tensors.sequence()<<", "<<input_tensors.audio_tensors.head()<<", "<<input_tensors.audio_tensors.dimension()<<"]"<<std::endl;
    auto audio_model = ImagebindAudioModel(config);
    audio_model.load(model_path);
    for (int step = 0; step < loop_times; step++) {
        auto result = audio_model({input_tensors.audio_tensors});
    }
    audio_model.profiling();
}