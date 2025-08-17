// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"
#include "mllm/utils/CPUArchHelper.hpp"
#include "driver.hpp"

#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
#include "schema/kai.hpp"
#endif

using mllm::Argparse;

int main(int argc, char** argv) {
  mllm::initializeContext();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& input_files = Argparse::add<std::string>("-i|--input").help("Input file path.").meta("FILE").positional();
  auto& config_files = Argparse::add<std::string>("-c|--config").help("config file path.");
  auto& input_file_version = Argparse::add<std::string>("-iv|--input_version").help("Input file version.");
  auto& output_files = Argparse::add<std::string>("-o|--output").help("Output file path.");

  Argparse::parse(argc, argv);

  if (help.isSet()) {
    Argparse::printHelp();
    return 0;
  }

  if (!input_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input file path provided");
    Argparse::printHelp();
    return -1;
  }

  if (!output_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No output file path provided");
    Argparse::printHelp();
    return -1;
  }

  if (!input_file_version.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No input file version provided");
    Argparse::printHelp();
    return -1;
  }

  if (!config_files.isSet()) {
    MLLM_ERROR_EXIT(mllm::ExitCode::kCoreError, "No config file path provided");
    Argparse::printHelp();
    return -1;
  }

  mllm::setRandomSeed(42);

  // Load input file and config file.
  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (input_file_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (input_file_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  }

  auto params = mllm::load(input_files.get(), file_version, mllm::kCPU);

  mllm::print("Model loaded.");
  mllm::print(params);

  auto config = mllm::ConfigFile(config_files.get());

  mllm::print("Config loaded.");
  mllm::print(config);

  // Quantize those params!
  QuantizeDriver qd(params, config);

// Register
#if defined(MLLM_HOST_ARCH_ARM64) || defined(MLLM_HOST_ARCH_ARM)
  qd.registerQuantizeImpl(QuantizeImpl_KAI_fp16_fp16_fp16p_mxk_kxn::create());
  qd.registerQuantizeImpl(QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_nxk::create());
  qd.registerQuantizeImpl(QuantizeImpl_KAI_f32_qai8dxp_qsi4c32p_mxk_kxn::create());
  qd.registerQuantizeImpl(QuantizeImpl_KAI_f16_qsi8d32p_qai4c32p_mxk_nxk::create());
#endif
  qd();

  mllm::print(params);
  // TODO
  // mllm::save(output_files.get(), params);

  mllm::shutdownContext();
}
