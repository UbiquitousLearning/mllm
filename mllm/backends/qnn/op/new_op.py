import os
import sys


code_hpp = """
#ifndef MLLM_QNNADD_H
#define MLLM_QNNADD_H

#include "QNNCommonOp.hpp"
namespace mllm {
class QNNAdd : public QNNCommonOp {
public:
    QNNAdd(Backend *bn, string opName);
    virtual ~QNNAdd() = default;
    virtual ErrorCode reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
    virtual ErrorCode setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) override;
};

class QNNAddCreator : public QNNBackend::Creator {
public:
    virtual Op *create(OpParam op_param, Backend *bn, string name) const {
        return new QNNAdd(bn, name);
    }
};

} // namespace mllm

#endif
"""

code_cpp = """
#include "QNNAdd.hpp"
#include "Types.hpp"
#include "QNNCommonOp.hpp"

namespace mllm {
QNNAdd::QNNAdd(Backend *bn, string opName) :
    QNNCommonOp(bn, opName) {
}

ErrorCode QNNAdd::reshape(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return NO_ERROR;
}

ErrorCode QNNAdd::setUp(vector<shared_ptr<Tensor>> inputs, vector<shared_ptr<Tensor>> outputs) {
    return graphAddNode(name(), "Add", inputs, outputs);
}
} // namespace mllm

"""

if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        print(
            "Usage: python new_op.py [op_name]\n   e.g. python new_op.py QNNXXX"
        )
        exit(1)

    op_name = args[1]
    dirname = op_name[:3]
    op_name_upper = op_name.upper()
    if dirname == "QNN":
        new_code_hpp = code_hpp.replace("QNNAdd", op_name)
        new_code_hpp = new_code_hpp.replace("QNNADD", op_name_upper)
        file_hpp = os.getcwd() + "/" + op_name + ".hpp"
        file = open(file_hpp, "w")
        file.write(new_code_hpp)
        file.close()
        new_code_cpp = code_cpp.replace("QNNAdd", op_name)
        new_code_cpp = new_code_cpp.replace("QNNADD", op_name_upper)
        file_hpp = os.getcwd() + "/" + op_name + ".cpp"
        file = open(file_hpp, "w")
        file.write(new_code_cpp)
        file.close()
    else:
        print("Only support CPUXXX now!")
        exit(1)
