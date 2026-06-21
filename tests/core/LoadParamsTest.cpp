#include "mllm/mllm.hpp"

using mllm::Argparse;
using mllm::load;
using mllm::print;

int main(int argc, char** argv) {
  mllm::initializeContext();
  auto& help = Argparse::add<bool>("-h|--help").help("Show help message");
  auto& input_files = Argparse::add<std::string>("-i|--input").help("Input file path").meta("FILE").positional();

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

  print(load(input_files.get()));
}
