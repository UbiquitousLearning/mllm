#include "mllm/mllm.hpp"

using mllm::Argparse;
using mllm::load;
using mllm::print;

int main(int argc, char** argv) {
  mllm::initializeContext();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& input_files = Argparse::add<std::string>("-i|--input").help("Input file path.").meta("FILE").positional();
  auto& input_files_version =
      Argparse::add<std::string>("-iv|--input_version").help("Input file version.").meta("FILE").positional();
  auto& specific_params = Argparse::add<std::string>("-n|--name").help("Specific param name.");

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

  mllm::ModelFileVersion file_version = mllm::ModelFileVersion::kV1;
  if (input_files_version.get() == "v1") {
    file_version = mllm::ModelFileVersion::kV1;
  } else if (input_files_version.get() == "v2") {
    file_version = mllm::ModelFileVersion::kV2;
  }

  if (specific_params.isSet()) {
    print(load(input_files.get(), file_version)->pull(specific_params.get()));
  } else {
    print(load(input_files.get(), file_version));
  }
}
