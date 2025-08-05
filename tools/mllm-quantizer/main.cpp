// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/mllm.hpp"

#include "driver.hpp"

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

  mllm::initializeContext();
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
  QuantizeDriver _({params}, config);

  mllm::save(output_files.get(), params);

  mllm::shutdownContext();
}