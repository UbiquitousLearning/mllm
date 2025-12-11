// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace mllm::qnn::aot {

enum class QcomHTPArch : uint32_t {
  NONE = 0,
  V68 = 68,
  V69 = 69,
  V73 = 73,
  V75 = 75,
  V79 = 79,
  V81 = 81,
};

enum QcomChipset : int {
  UNKNOWN_SM = 0,
  SA8295 = 39,
  SM8350 = 35,
  SM8450 = 36,
  SM8475 = 42,
  SM8550 = 43,
  SM8650 = 57,
  SM8750 = 69,
  SM8850 = 87,
  SSG2115P = 46,
  SSG2125P = 58,
  SXR1230P = 45,
  SXR2230P = 53,
  SXR2330P = 75,
  QCS9100 = 77,
  SAR2230P = 95,
  SA8255 = 52,
  SW6100 = 96,
};

enum QcomTryBestPerformance : int {
  kHtpDefault = 0,
  kHtpSustainedHighPerformance,
  kHtpBurst,
  kHtpHighPerformance,
  kHtpPowerSaver,
  kHtpLowPowerSaver,
  kHtpHighPowerSaver,
  kHtpLowBalanced,
  kHtpBalanced,
};

//  Protection Domain Session
enum QcomSecurityPDSession : int {
  kHtpUnsignedPd = 0,
  kHtpSignedPd,
};

struct QcomTargetMachine {
  QcomChipset soc_htp_chipset;
  QcomHTPArch soc_htp_arch;
  QcomTryBestPerformance soc_htp_performance;
  QcomSecurityPDSession soc_htp_security_pd_session;
};

}  // namespace mllm::qnn::aot
