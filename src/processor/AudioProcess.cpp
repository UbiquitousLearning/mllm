//
// Created by ey on 24-1-9.
//

#include "AudioProcess.hpp"
#include <iomanip>
#include <utility>
#include <iostream>
#include <memory>
#include <vector>
#include "wenet_audio/params.h"
#include "wenet_audio/wav.h"
#include "wenet_audio/feature_pipeline.h"

float* waveClip(const float* data_, int start, int end, int channel)
{
    std::vector<float> even_elements;
    // std::vector<float> odd_elements;

    for (int i = start; i < end; ++i) {
        if (i % channel == 0) {  //选择第一个channel TODO：torchaudio/compliance/kaldi.py #L125 _get_waveform_and_window_properties()
            even_elements.push_back(data_[i]);
        }
    }
    auto data_new = new float[(end-start)/channel];
    std::copy(even_elements.begin(), even_elements.end(), data_new);
    // auto data_new = new float[end-start];
    // for (int i = start; i < end; i++)
    // {
    //     data_new[i-start] = data_[i];;
    // }
    return data_new;
}
std::vector<std::vector<float>> readFeats(const std::shared_ptr<wenet::FeaturePipeline>& feature_pipeline, const int num_frames_, const int feature_dim_)
{
    bool end_flag = false;
    std::vector<std::vector<float>> chunk_feats;
    while (!end_flag) {
        //这里主要实现的是，读取一段音频，对音频进行每67个frame一次送入forward，
        if (!feature_pipeline->Read(num_frames_, &chunk_feats)) //说明feat结束，没有获取67个frame数据，则自动补0
        {
            int padding_len = num_frames_ - chunk_feats.size();
            std::vector<float> zero_vector(feature_dim_, 0);
            for (int i = 0; i < padding_len; i++) {
                chunk_feats.push_back(zero_vector);
            }
            end_flag = true;
        }
    }
    return chunk_feats;
}
std::vector<std::vector<float>> transpose(std::vector<std::vector<float>> chunk_feats)
{
    std::vector<std::vector<float>> transposed(chunk_feats[0].size(), std::vector<float>(chunk_feats.size()));
    for (size_t i = 0; i < chunk_feats.size(); ++i)
        for (size_t j = 0; j < chunk_feats[i].size(); ++j)
            transposed[j][i] = chunk_feats[i][j];
    return transposed;
}
void printFeats(std::vector<std::vector<float>> chunk_feats)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(4);
    for (size_t i = 0; i < chunk_feats.size(); ++i) {
        for (size_t j = 0; j < chunk_feats[i].size(); ++j) {
            std::cout << chunk_feats[i][j] << ",";
        }
        std::cout << std::endl;
    }
}
void printAllClips( std::vector<std::vector<std::vector<float>>> all_clips)
{
    for (auto all_clip : all_clips)
    {
        printFeats(all_clip);
        std::cout<< "======================================" << std::endl;
    }
}

void Normalize(std::vector<std::vector<float>>& chunk_feats, const float mean, const float std)
{
    for (auto & chunk_feat : chunk_feats) {
        for (float & j : chunk_feat) {
            j = (j - mean) / std;
        }
    }
}

void printdata_(const float *data_, int num_data)
{
    std::cout << std::fixed;
    std::cout << std::setprecision(8);
    for (int i = 0; i < num_data; i++)
    {
        std::cout<<data_[i]<<" ";
    }
    std::cout<<std::endl;
    std::cout<<num_data<<std::endl;
}

std::vector<std::vector<std::vector<std::vector<float>>>> ProcessWAV(std::vector<std::string> waves){
    auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
    auto feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
    std::vector<std::vector<std::vector<std::vector<float>>>> output_audios;
    for (auto &wav: waves)
    {
        wenet::WavReader wav_reader(wav);
        wav_reader.rescale();
        // wav_reader.print();
        //TODO: 1. resample:=sample_rate  /imagebind/data.py #L137~140
        //wav_reader.resample();
        //TODO: 2. get_clip_timepoints  /imagebind/data.py #L141~143
        auto waveform_size = wav_reader.num_sample();
        auto sample_rate = 16000;
        // std::vector<std::pair<int, int>> clip_timepoints = {{0, 32000}, {28856, 60856}, {57712, 89712}};
        std::vector<std::pair<int, int>> clip_timepoints = {{0, 32000}, {4882, 36882}, {9764, 41764}}; //for ../dog_audio_16k.wav
        if( wav.find("car") != std::string::npos)
            clip_timepoints = {{0, 32000}, {24000, 56000}, {48000, 80000}};
        if( wav.find("bird") != std::string::npos)
            clip_timepoints = {{0, 32000}, {24000, 56000}, {48000, 80000}};
        //TODO: get_clip_timepoints  END  /imagebind/data.py #L141~143
        std::vector<std::vector<std::vector<float>>> all_clips;
        for (auto clip_timepoint : clip_timepoints)
        {
            const int clip_start = clip_timepoint.first * wav_reader.num_channel();
            const int clip_end = clip_timepoint.second * wav_reader.num_channel();
            const auto datac = waveClip(wav_reader.data(), clip_start, clip_end, wav_reader.num_channel());
            // printdata_(datac, (clip_end - clip_start)/wav_reader.num_channel());
            const int datac_num_sample = (clip_end - clip_start)/wav_reader.num_channel();
            feature_pipeline->AcceptWaveform(std::vector<float>(datac, datac + datac_num_sample));
            const int num_frames_ = 204;
            const int feature_dim_ = 128;
            const auto chunk_feats = readFeats(feature_pipeline, num_frames_, feature_dim_);
            auto outfeats = transpose(chunk_feats);
            Normalize(outfeats, -4.268, 9.138);
            all_clips.push_back(outfeats);
            //PRINT
            // printFeats(outfeats);
            // std::cout<< "======================================" << std::endl;
        }
        // printAllClips(all_clips); // all_clips 对应/imagebind/data.py 中  load_and_transform_audio_data的all_clips #L160
        output_audios.push_back(all_clips);
    }
    return output_audios;
}
