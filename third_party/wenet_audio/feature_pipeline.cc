// Copyright (c) 2017 Personal (Binbin Zhang)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "wenet_audio/feature_pipeline.h"

#include <algorithm>
#include <utility>

namespace wenet {

FeaturePipeline::FeaturePipeline(const FeaturePipelineConfig& config)
    : config_(config),
      feature_dim_(config.num_bins),
      fbank_(config.num_bins, config.sample_rate, config.frame_length,
             config.frame_shift),
      num_frames_(0){}
      //input_finished_(false) {}

void FeaturePipeline::AcceptWaveform(const std::vector<float>& wav) {
  std::vector<std::vector<float>> feats;
  int num_frames = fbank_.Compute(wav, &feats);
  for (size_t i = 0; i < feats.size(); ++i) {
    feature_queue_.push(std::move(feats[i]));
  }
  num_frames_ += num_frames;

}

bool FeaturePipeline::ReadOne(std::vector<float>* feat) {
  if (!feature_queue_.empty()) {
    *feat = std::move(feature_queue_.front());
    feature_queue_.pop();
    return true;
  } else {
  return false; 
  }  
}

bool FeaturePipeline::Read(int num_frames,
                           std::vector<std::vector<float>>* feats) {
  feats->clear();
  std::vector<float> feat;
  while (feats->size() < num_frames) {
    if (ReadOne(&feat)) {
      feats->push_back(std::move(feat));
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace wenet
