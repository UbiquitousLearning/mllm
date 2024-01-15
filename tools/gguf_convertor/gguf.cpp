#include "gguf.hpp"
#include "ParamWriter.hpp"
#include <string>
// gguf input_file outfile
int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " input_file outfile" << std::endl;
        return 1;
    }
    std::string input_file(argv[1]);
    std::string output_file(argv[2]);
    auto *writer = new ParamWriter(output_file);
    from_gguf(input_file, writer);
    return 0;
}
