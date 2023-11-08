import os
import sys
assert(os.getcwd().split('/')[-1] == 'backends')


code_hpp = '''
#ifndef MLLM_CPUABC_H
#define MLLM_CPUABC_H

#include "Op.hpp"
#include "CPUBackend.hpp"

namespace mllm {

class CPUAbc final : public Op {
public:
    CPUAbc(Backend *bn, string opName, bool multiThread);
    virtual ~CPUAbc() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode load(ParamLoader &loader) override;
    virtual ErrorCode execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;

private:
    bool support_multi_thread_ = false;
};

class CPUAbcCreator : public CPUBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new CPUAbc(bn, name, false);
    }
};

} // namespace mllm

#endif // MLLM_CPUABC_H
'''

code_cpp = '''

#include "CPUAbc.hpp"

namespace mllm {

CPUAbc::CPUAbc(Backend *bn,  string opName, bool multiThread) :
    Op(bn, opName) {
}

ErrorCode CPUAbc::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUAbc  reshape" << std::endl;
    return Op::reshape(inputs, outputs);
}

ErrorCode CPUAbc::load(ParamLoader &loader) {
    //std::cout<<name() << "  CPUAbc load" << std::endl;
    return Op::load(loader);
}

ErrorCode CPUAbc::execute(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUAbc()" << std::endl;
    return Op::execute(inputs, outputs);
}

ErrorCode CPUAbc::free(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    //std::cout<<name() << "  CPUAbc() free" << std::endl;
    return Op::free(inputs, outputs);
}
} // namespace mllm

'''

if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Usage: python build_new_op.py [op_name]\n   e.g. python build_new_op.py CPUXXX')
        exit(1)
    op_name = args[1]
    dirname = op_name[:3]
    op_name_upper = op_name.upper()
    if dirname == 'CPU':
        new_code_hpp = code_hpp.replace("CPUAbc", op_name)
        new_code_hpp = new_code_hpp.replace("CPUABC", op_name_upper)
        file_hpp = os.getcwd() + '/cpu/' + op_name + ".hpp"
        file = open(file_hpp, "w")
        file.write(new_code_hpp)
        file.close()
        new_code_cpp = code_cpp.replace("CPUAbc", op_name)
        new_code_cpp = new_code_cpp.replace("CPUABC", op_name_upper)
        file_hpp = os.getcwd() + '/cpu/' + op_name + ".cpp"
        file = open(file_hpp, "w")
        file.write(new_code_cpp)
        file.close()
    else:
        print('Only support CPUXXX now!')
        exit(1)