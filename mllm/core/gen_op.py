import os
from string import Template

HEADER_TEMPLATE = Template(
    """// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct ${OpName}OpOptions : public BaseOpOptions<${OpName}OpOptions> {};

class ${OpName}Op : public BaseOp {
 public:
  explicit ${OpName}Op(const ${OpName}OpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

 protected:
  ${OpName}OpOptions options_;
};

}  // namespace mllm::aops
"""
)

CPP_TEMPLATE = Template(
    """// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#include "mllm/core/aops/${OpName}Op.hpp"
#include "mllm/core/BaseOp.hpp"
#include "mllm/core/Tensor.hpp"
#include "mllm/utils/Common.hpp"
#include "mllm/compile/ir/linalg/Op.hpp"

namespace mllm::aops {

${OpName}Op::${OpName}Op(const ${OpName}OpOptions& options) : BaseOp(OpTypes::k${OpName}), options_(options) {}

void ${OpName}Op::load(const ParameterFile::ptr_t& ploader) { MLLM_EMPTY_SCOPE; }

void ${OpName}Op::trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
  MLLM_EMPTY_SCOPE;
}

void ${OpName}Op::forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ${OpName}Op::reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

void ${OpName}Op::setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) { MLLM_EMPTY_SCOPE; }

}  // namespace mllm::aops
"""
)


def generate_op_files(op_name, out_dir="./aops"):
    op = op_name[0].upper() + op_name[1:]
    header_content = HEADER_TEMPLATE.substitute(OpName=op)
    cpp_content = CPP_TEMPLATE.substitute(OpName=op)

    hpp_name = os.path.join(out_dir, f"{op}Op.hpp")
    cpp_name = os.path.join(out_dir, f"{op}Op.cpp")

    with open(hpp_name, "w") as f:
        f.write(header_content)
    with open(cpp_name, "w") as f:
        f.write(cpp_content)

    print("Generated files:")
    print("  -", hpp_name)
    print("  -", cpp_name)
    print("You need to add OpType in OpTypes.hpp for", op)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gen_op.py OpName")
    else:
        generate_op_files(sys.argv[1])
