// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "QnnTargetMachine.hpp"
#include <string>

namespace mllm::qnn::aot {

// Parse QcomTargetMachine from JSON string
// JSON format:
// {
//     "target_machine": {
//         "htp_arch": "V81",
//         "htp_chipset": "SM8850",
//         "htp_try_best_performance": "HtpBurst",
//         "htp_security_pd_session": "HtpSignedPd",
//         "htp_vtcm_capability_in_mb": 8
//     }
// }
QcomTargetMachine parseQcomTargetMachineFromJSON(const std::string& json_str);

// Parse QcomTargetMachine from JSON file path
QcomTargetMachine parseQcomTargetMachineFromJSONFile(const std::string& file_path);

}  // namespace mllm::qnn::aot
