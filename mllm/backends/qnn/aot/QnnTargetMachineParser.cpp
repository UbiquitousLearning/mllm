// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

#include "QnnTargetMachineParser.hpp"

namespace mllm::qnn::aot {

// Helper function to parse HTP architecture from string
QcomHTPArch parseHTPArch(const std::string& arch_str) {
  if (arch_str == "NONE") return QcomHTPArch::NONE;
  if (arch_str == "V68") return QcomHTPArch::V68;
  if (arch_str == "V69") return QcomHTPArch::V69;
  if (arch_str == "V73") return QcomHTPArch::V73;
  if (arch_str == "V75") return QcomHTPArch::V75;
  if (arch_str == "V79") return QcomHTPArch::V79;
  if (arch_str == "V81") return QcomHTPArch::V81;
  throw std::invalid_argument("Unknown HTP architecture: " + arch_str);
}

// Helper function to parse chipset from string
QcomChipset parseChipset(const std::string& chipset_str) {
  if (chipset_str == "UNKNOWN_SM") return UNKNOWN_SM;
  if (chipset_str == "SA8295") return SA8295;
  if (chipset_str == "SM8350") return SM8350;
  if (chipset_str == "SM8450") return SM8450;
  if (chipset_str == "SM8475") return SM8475;
  if (chipset_str == "SM8550") return SM8550;
  if (chipset_str == "SM8650") return SM8650;
  if (chipset_str == "SM8750") return SM8750;
  if (chipset_str == "SM8850") return SM8850;
  if (chipset_str == "SM8845") return SM8845;
  if (chipset_str == "SSG2115P") return SSG2115P;
  if (chipset_str == "SSG2125P") return SSG2125P;
  if (chipset_str == "SXR1230P") return SXR1230P;
  if (chipset_str == "SXR2230P") return SXR2230P;
  if (chipset_str == "SXR2330P") return SXR2330P;
  if (chipset_str == "QCS9100") return QCS9100;
  if (chipset_str == "SAR2230P") return SAR2230P;
  if (chipset_str == "SA8255") return SA8255;
  if (chipset_str == "SW6100") return SW6100;
  throw std::invalid_argument("Unknown chipset: " + chipset_str);
}

// Helper function to parse performance mode from string
QcomTryBestPerformance parsePerformance(const std::string& perf_str) {
  if (perf_str == "HtpDefault") return kHtpDefault;
  if (perf_str == "HtpSustainedHighPerformance") return kHtpSustainedHighPerformance;
  if (perf_str == "HtpBurst") return kHtpBurst;
  if (perf_str == "HtpHighPerformance") return kHtpHighPerformance;
  if (perf_str == "HtpPowerSaver") return kHtpPowerSaver;
  if (perf_str == "HtpLowPowerSaver") return kHtpLowPowerSaver;
  if (perf_str == "HtpHighPowerSaver") return kHtpHighPowerSaver;
  if (perf_str == "HtpLowBalanced") return kHtpLowBalanced;
  if (perf_str == "HtpBalanced") return kHtpBalanced;
  throw std::invalid_argument("Unknown performance mode: " + perf_str);
}

// Helper function to parse security PD session from string
QcomSecurityPDSession parseSecurityPDSession(const std::string& security_str) {
  if (security_str == "HtpUnsignedPd") return kHtpUnsignedPd;
  if (security_str == "HtpSignedPd") return kHtpSignedPd;
  throw std::invalid_argument("Unknown security PD session: " + security_str);
}

QcomTargetMachine parseQcomTargetMachineFromJSON(const std::string& json_str) {
  try {
    auto json = nlohmann::json::parse(json_str);

    if (!json.contains("target_machine")) { throw std::invalid_argument("JSON must contain 'target_machine' field"); }

    auto tm = json["target_machine"];

    QcomTargetMachine result;
    result.soc_htp_arch = parseHTPArch(tm["htp_arch"].get<std::string>());
    result.soc_htp_chipset = parseChipset(tm["htp_chipset"].get<std::string>());
    result.soc_htp_performance = parsePerformance(tm["htp_try_best_performance"].get<std::string>());
    result.soc_htp_security_pd_session = parseSecurityPDSession(tm["htp_security_pd_session"].get<std::string>());
    result.soc_htp_vtcm_total_memory_size = tm["htp_vtcm_capability_in_mb"].get<uint32_t>();

    return result;
  } catch (const nlohmann::json::exception& e) { throw std::invalid_argument(std::string("JSON parsing error: ") + e.what()); }
}

QcomTargetMachine parseQcomTargetMachineFromJSONFile(const std::string& file_path) {
  std::ifstream file(file_path);
  if (!file.is_open()) { throw std::runtime_error("Cannot open file: " + file_path); }

  std::stringstream buffer;
  buffer << file.rdbuf();

  return parseQcomTargetMachineFromJSON(buffer.str());
}

}  // namespace mllm::qnn::aot
