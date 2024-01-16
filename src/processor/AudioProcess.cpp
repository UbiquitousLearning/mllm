//
// Created by Rongjie Yi on 24-1-9.
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

#include <algorithm>


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


std::pair<int, std::vector<std::vector<std::vector<float>>>> get_sinc_resample_kernel(
    int new_freq, int gcd, int orig_freq)
{
    // int orig_freq = sample_rate_;
    float lowpass_filter_width = 6;
    float rolloff = 0.99;
    std::string resampling_method = "sinc_interp_hann";

    orig_freq = int(orig_freq) / gcd;
    new_freq = int(new_freq) / gcd;
    // std::cout << "orig_freq: " << orig_freq << std::endl;

    if (lowpass_filter_width <= 0)
        std::cout << "lowpass_filter_width must be positive" << std::endl;
    float base_freq = std::min(orig_freq, new_freq);
    base_freq *= rolloff;
    // std::cout << "base_freq: " << base_freq << std::endl;

    int width = ceil(lowpass_filter_width * orig_freq / base_freq);
    // std::cout << "width: " << width << std::endl;

    // 实现 idx = torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=device)[None, None] / orig_freq
    std::vector<float> idx(orig_freq + 2 * width);
    for (int i = 0; i < orig_freq + 2 * width; i++)
    {
        idx[i] = float(-width + i) / float(orig_freq);
        // std::cout << "idx[" << i << "]: " << idx[i] << std::endl;
    }

    // 实现 t = torch.arange(0, -new_freq, -1, dtype=dtype, device=device)[:, None, None] / new_freq + idx
    std::vector<float> t(new_freq * idx.size());
    float t_temp = 0;
    for (int i = 0; i < new_freq; i++)
    {
        for (int j = 0; j < idx.size(); j++)
        {
            t[i * idx.size() + j] += t_temp;
            // std::cout << "t[" << i * idx.size() + j << "]: " << t[i * idx.size() + j] << std::endl;
        }
        t_temp -= 1;
    }

    for (int i = 0; i < t.size(); i++)
    {
        t[i] = float(t[i]) / float(new_freq);
        // std::cout << "t[" << i << "]: " << t[i] << std::endl;
    }

    // 将 t 和 idx 相加
    for (int i = 0; i < new_freq; i++)
        for (int j = 0; j < idx.size(); j++)
        {
            t[i * idx.size() + j] += idx[j];
            // std::cout << "t[" << i * idx.size() + j << "]: " << t[i * idx.size() + j] << std::endl;
        }

    // Print idx
    // std::cout << "idx: ";
    // for (auto& value : idx)
    // {
    //     std::cout << value << " ";
    // }
    // std::cout << std::endl;

    // Print t
    // std::cout << "t: ";
    // for (double value : t)
    // {
    //     // std::cout << value << " ";
    // }

    for (int i = 0; i < t.size(); i++)
    {
        t[i] = float(t[i]) * float(base_freq);
        // std::cout << "t[" << i << "]: " << t[i] << std::endl;
    }

    for (auto& value : t)
    {
        value = std::max(-lowpass_filter_width, std::min(value, lowpass_filter_width));
    }

    // Print t
    // std::cout << "t: ";
    // for (double value : t)
    // {
    //     std::cout << value << " ";
    // }

    std::vector<float> window(new_freq * idx.size());
    for (int i = 0; i < t.size(); i++)
    {
        window[i] = std::pow(cosf(t[i] * M_PI / lowpass_filter_width / 2), 2);
        // std::cout << "window[" << i << "]: " << window[i] << std::endl;
    }

    for (int i = 0; i < t.size(); i++)
    {
        t[i] = t[i] * M_PI;
        // std::cout << "t[" << i << "]: " << t[i] << std::endl;
    }

    float scale = base_freq / orig_freq;

    std::vector<float> kernels(t.size());
    std::transform(t.begin(), t.end(), kernels.begin(), [](double val)
    {
        return (val == 0) ? 1.0 : std::sin(val) / val;
    });

    std::vector<std::vector<std::vector<float>>> result;
    result.resize(new_freq);
    for (auto& res : result)
    {
        res.resize(1);
        res[0].resize(kernels.size() / new_freq);
    }
    for (int i = 0; i < kernels.size(); i++)
    {
        kernels[i] = kernels[i] * window[i] * scale;
        result[i / (kernels.size() / new_freq)][0][i % (kernels.size() / new_freq)] = kernels[i];
        // std::cout << "kernels[" << i << "]: " << kernels[i] << " " << result[i / (kernels.size() / new_freq)][0][i % (kernels.size() / new_freq)] << " " << i / (kernels.size() / new_freq) << " " << i % (kernels.size() / new_freq) << std::endl;
    }
    // printAllClips(result);
    return std::make_pair(width, result);
}

std::vector<std::vector<float>> wav_pad(std::vector<std::vector<float>> orig_wav, int pad_left, int pad_right)
{
    std::vector<std::vector<float>> result;
    result.resize(orig_wav.size());
    for (auto &re : result)
    {
        re.resize(orig_wav[0].size() + pad_left + pad_right);
    }
    for (int i = 0; i < orig_wav.size(); i++)
    {
        memset(result[i].data(), 0, pad_left * sizeof(float));
        memcpy(result[i].data() + pad_left, orig_wav[i].data(), orig_wav[i].size() * sizeof(float));
        memset(result[i].data() + pad_left + orig_wav[i].size(), 0, pad_right * sizeof(float));
    }
    return result;
}
std::vector<std::vector<std::vector<float>>> conv1d(std::vector<std::vector<float>> input, std::vector<std::vector<std::vector<float>>> kernel, int stride)
{
    int batch_size = input.size();
    int input_dim = input[0].size();
    int out_channels = kernel.size();
    int in_channels = kernel[0].size();
    int kernel_size = kernel[0][0].size();

    // 计算输出维度
    int output_dim = (input_dim - kernel_size) / stride + 1;

    // 初始化输出张量
    std::vector<std::vector<std::vector<float>>> output(
        batch_size, std::vector<std::vector<float>>(
            out_channels, std::vector<float>(output_dim, 0.0)
        )
    );

    // 进行卷积计算
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int i = 0; i < output_dim; ++i) {
                // 计算卷积的结果
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int input_index = i * stride + k;
                        int output_index = i;
                        assert( input_index < input_dim && output_index < output_dim);
                        output[b][oc][i] += input[b][input_index] * kernel[oc][ic][k];
                    }
                }
            }
        }
    }


    return output;
}

std::vector<std::vector<float>> conv1d_and_trans_and_viw(std::vector<std::vector<float>> input, std::vector<std::vector<std::vector<float>>> kernel, int stride)
{
    int batch_size = input.size();
    int input_dim = input[0].size();
    int out_channels = kernel.size();
    int in_channels = kernel[0].size();
    int kernel_size = kernel[0][0].size();

    // 计算输出维度
    int output_dim = (input_dim - kernel_size) / stride + 1;

    // 初始化输出张量
    std::vector<std::vector<float>> output(
        batch_size, std::vector<float>( out_channels * output_dim, 0.0)
    );

    // 进行卷积计算
    for (int b = 0; b < batch_size; ++b) {
        for (int oc = 0; oc < out_channels; ++oc) {
            for (int i = 0; i < output_dim; ++i) {
                // 计算卷积的结果
                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int k = 0; k < kernel_size; ++k) {
                        int input_index = i * stride + k;
                        int output_index = i;
                        assert( input_index < input_dim && output_index < output_dim);
                        output[b][i * out_channels+oc] += input[b][input_index] * kernel[oc][ic][k];
                    }
                }
            }
        }
    }


    return output;
}
std::vector<float> cut_and_trans(std::vector<std::vector<float>> wav, int target_length)
{
    std::vector<float> result(target_length*wav.size());
    for (int i = 0; i < target_length*wav.size(); ++i)
    {
        auto dim_b = i % wav.size();
        auto dim_d = i / wav.size();
        result[i] = wav[dim_b][dim_d];
    }
    return result;
}
std::vector<float> apply_sinc_resample_kernel(std::vector<std::vector<float>> orig_wav, int orig_freq, int new_freq,
                                               int gcd, std::vector<std::vector<std::vector<float>>> kernel,
                                               int width){
    auto length = orig_wav[0].size() ;
    orig_freq = int(orig_freq) / gcd;
    new_freq = int(new_freq) / gcd;
    std::vector<float> result;
    orig_wav = wav_pad(orig_wav, width, width+orig_freq);
    auto resample_wav = conv1d_and_trans_and_viw(orig_wav, std::move(kernel), orig_freq);

    int target_length = static_cast<int>(std::ceil(static_cast<double>(new_freq) * length/ orig_freq));
    result = cut_and_trans(resample_wav, target_length);



    // std::cout << std::fixed;
    // std::cout << std::setprecision(4);
    // for (size_t j = 0; j < result.size(); ++j)
    // {
    //     std::cout << result[j] << ",";
    // }
    // std::cout << std::endl;
    // std::cout <<result.size()/2<< std::endl;
    return result;
}

std::vector<float> resample(std::vector<std::vector<float>> orig_wav, int new_freq, int orig_freq)
{
    std::vector<float> resampled;
    // int orig_freq = sample_rate_;
    if (new_freq == orig_freq)
    {
        return resampled;
    }
    int gcd = std::__gcd(new_freq, orig_freq);
    // std::cout << "gcd: " << gcd << std::endl;

    auto width_kernel = get_sinc_resample_kernel(new_freq, gcd, orig_freq);
    int width = width_kernel.first;
    auto kernel = width_kernel.second;

    resampled =  apply_sinc_resample_kernel(orig_wav, orig_freq, new_freq, gcd, kernel, width);
    return resampled;
}

std::vector<std::vector<float>> get_wav_data(const float* wavdata, int wavdata_sample, int wavdata_channel)
{
    std::vector<std::vector<float>> result;
    result.resize(wavdata_channel);
    for (auto &data : result)
    {
        data.resize(wavdata_sample);
    }
    for (int i = 0; i < wavdata_sample*wavdata_channel; i++)
    {
        result[i % wavdata_channel][i / wavdata_channel] = wavdata[i];
        // std::cout<<wavdata[i]<<" "<<i % wavdata_channel<<" "<<i / wavdata_channel<<std::endl;
    }
    // std::cout<<std::endl;

    // printFeats(result);

    return result;
}


std::vector<std::vector<std::vector<std::vector<float>>>> ProcessWAV(std::vector<std::string> waves, int resample_rate){
    auto feature_config = wenet::InitFeaturePipelineConfigFromFlags();
    auto feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
    std::vector<std::vector<std::vector<std::vector<float>>>> output_audios;
    for (auto &wav: waves)
    {
        wenet::WavReader wav_reader(wav);
        wav_reader.rescale();
        // wav_reader.print();
        //TODO: 1. resample:=sample_rate  /imagebind/data.py #L137~140
        auto wavdata = wav_reader.data();
        auto wavdata_sample = wav_reader.num_sample();
        auto wavdata_channel = wav_reader.num_channel();
        auto origin_sample_rate = wav_reader.sample_rate();
        auto wavdata_size = wavdata_sample*wavdata_channel;
        std::vector<std::vector<float>> wav_data = get_wav_data(wavdata, wavdata_sample, wavdata_channel);
        auto resampled = resample(wav_data, resample_rate, origin_sample_rate);

        //TODO: 2. get_clip_timepoints  /imagebind/data.py #L141~143
        auto waveform_size = wav_reader.num_sample();
        auto sample_rate = resample_rate;
        // std::vector<std::pair<int, int>> clip_timepoints = {{0, 32000}, {28856, 60856}, {57712, 89712}};
        std::vector<std::pair<int, int>> clip_timepoints = {{0, 32000}, {4882, 36882}, {9764, 41764}}; //for ../dog_audio.wav
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
            const float *datac;
            if(origin_sample_rate == resample_rate)
            {
                datac = waveClip(wav_reader.data(), clip_start, clip_end, wav_reader.num_channel());
            }else
            {
                datac = waveClip(resampled.data(), clip_start, clip_end, wav_reader.num_channel());
            }
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
