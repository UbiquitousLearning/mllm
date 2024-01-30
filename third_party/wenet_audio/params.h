// Copyright 2021 Mobvoi Inc. All Rights Reserved.
// Author: binbinzhang@mobvoi.com (Binbin Zhang)
//         di.wu@mobvoi.com (Di Wu)

#ifndef DECODER_PARAMS_H_
#define DECODER_PARAMS_H_

#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "wenet_audio/feature_pipeline.h"


namespace wenet {

std::shared_ptr<FeaturePipelineConfig> InitFeaturePipelineConfigFromFlags() {
  auto feature_config = std::make_shared<FeaturePipelineConfig>(
      128, 16000);
  return feature_config;
  }

}
#endif
