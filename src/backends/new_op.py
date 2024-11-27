import os
import sys

assert os.getcwd().split("/")[-1] == "backends"


code_hpp = """
#ifndef MLLM_CPUABC_H
#define MLLM_CPUABC_H

#include "Op.hpp"
#include "../CPUBackend.hpp"

namespace mllm {

class CPUAbc final : public Op {
public:
    CPUAbc(Backend *bn, string opName, int threadCount);
    ~CPUAbc() override = default;
    ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode load(AbstructLoader &loader) override;
    ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    int thread_count = 4;
};

class CPUAbcCreator : public CPUBackend::Creator {
public:
    Op *create(OpParam op_param, Backend *bn, string name, int threadCount) const override {
        return new CPUAbc(bn, name, threadCount);
    }
};

} // namespace mllm

#endif // MLLM_CPUABC_H
"""

code_cpp = """
#include "CPUAbc.hpp"

namespace mllm {

CPUAbc::CPUAbc(Backend *bn,  string opName, int threadCount) : thread_count(threadCount), 
    Op(bn, opName) {
}

ErrorCode CPUAbc::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUAbc::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::execute(inputs, outputs);
}

ErrorCode CPUAbc::load(AbstructLoader &loader) {
    return Op::load(loader);
}

ErrorCode CPUAbc::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::free(inputs, outputs);
}

ErrorCode CPUAbc::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return Op::setUp(inputs, outputs);
}
} // namespace mllm

"""

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print(
            "Usage: python build_new_op.py [op_name]\n   e.g. python build_new_op.py CPUXXX"
        )
        exit(1)
    op_name = args[1]
    dirname = op_name[:3]
    op_name_upper = op_name.upper()
    if dirname == "CPU":
        new_code_hpp = code_hpp.replace("CPUAbc", op_name)
        new_code_hpp = new_code_hpp.replace("CPUABC", op_name_upper)
        file_hpp = os.getcwd() + "/cpu/op/" + op_name + ".hpp"
        file = open(file_hpp, "w")
        file.write(new_code_hpp)
        file.close()
        new_code_cpp = code_cpp.replace("CPUAbc", op_name)
        new_code_cpp = new_code_cpp.replace("CPUABC", op_name_upper)
        file_hpp = os.getcwd() + "/cpu/op/" + op_name + ".cpp"
        file = open(file_hpp, "w")
        file.write(new_code_cpp)
        file.close()
    else:
        print("Only support CPUXXX now!")
        exit(1)
