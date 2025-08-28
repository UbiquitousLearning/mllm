/**
 * @file ReduceOps.hpp
 * @author chenghua wang (chenghua.wang.edu@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-08-01
 *
 */
#pragma once

#include "mllm/core/BaseOp.hpp"
#include "mllm/core/ParameterFile.hpp"

namespace mllm::aops {

struct ReduceMaxOpOptions : public BaseOpOptions<ReduceMaxOpOptions> {
  int32_t dim = 0x7fffffff;
  bool keep_dim = false;
};

class ReduceMaxOp : public BaseOp {
 public:
  explicit ReduceMaxOp(const ReduceMaxOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const ReduceMaxOpOptions& options() const { return options_; }

 protected:
  ReduceMaxOpOptions options_;
};

struct ReduceMinOpOptions : public BaseOpOptions<ReduceMinOpOptions> {
  int32_t dim = -1;
  bool keep_dim = false;
};

class ReduceMinOp : public BaseOp {
 public:
  explicit ReduceMinOp(const ReduceMinOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const ReduceMinOpOptions& options() const { return options_; }

 protected:
  ReduceMinOpOptions options_;
};

struct ReduceSumOpOptions : public BaseOpOptions<ReduceSumOpOptions> {
  int32_t dim = -1;
  bool keep_dim = false;
};

class ReduceSumOp : public BaseOp {
 public:
  explicit ReduceSumOp(const ReduceSumOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const ReduceSumOpOptions& options() const { return options_; }

 protected:
  ReduceSumOpOptions options_;
};

struct MeanOpOptions : public BaseOpOptions<MeanOpOptions> {
  int32_t dim = -1;
  bool keep_dim = false;
};

class MeanOp : public BaseOp {
 public:
  explicit MeanOp(const MeanOpOptions& cargo);

  void load(const ParameterFile::ptr_t& ploader) override;

  void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

  inline const MeanOpOptions& options() const { return options_; }

 protected:
  MeanOpOptions options_;
};

}  // namespace mllm::aops
