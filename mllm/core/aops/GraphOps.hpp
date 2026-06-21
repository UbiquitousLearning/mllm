// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct GraphBeginOpOptions : public BaseOpOptions<GraphBeginOpOptions> {
  std::string graph_name;
};

class GraphBeginOp : public BaseOp {
 public:
  explicit GraphBeginOp(const GraphBeginOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const GraphBeginOpOptions& options() const { return options_; }

 protected:
  GraphBeginOpOptions options_;
};

struct GraphEndOpOptions : public BaseOpOptions<GraphEndOpOptions> {
  std::string graph_name;
};

class GraphEndOp : public BaseOp {
 public:
  explicit GraphEndOp(const GraphEndOpOptions& options);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const GraphEndOpOptions& options() const { return options_; }

 protected:
  GraphEndOpOptions options_;
};

}  // namespace mllm::aops