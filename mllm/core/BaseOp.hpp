// Copyright (c) MLLM Team.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <cstddef>
#include <vector>

#include "mllm/core/Tensor.hpp"
#include "mllm/core/OpTypes.hpp"
#include "mllm/core/DataTypes.hpp"
#include "mllm/core/DeviceTypes.hpp"
#include "mllm/core/ParameterFile.hpp"

#ifdef MLLM_ENABLE_PY_MLLM
#include <typeindex>
#include <type_traits>
#endif

namespace mllm {

template<typename DerivedT>
class BaseOpOptions {
 public:
  BaseOpOptions() = default;
  BaseOpOptions(size_t inputs_len, size_t outputs_len, DataTypes default_dtype = kFloat32)
      : inputs_dtypes_(inputs_len, default_dtype), outputs_dtypes_(outputs_len, default_dtype) {}

  inline DerivedT& setInputsDtype(size_t pos, DataTypes dtype) {
    if (pos >= inputs_dtypes_.size()) { inputs_dtypes_.resize(pos + 1, kFloat32); }
    inputs_dtypes_[pos] = dtype;
    return *static_cast<DerivedT*>(this);
  }

  inline DerivedT& setOutputsDtype(size_t pos, DataTypes dtype) {
    if (pos >= outputs_dtypes_.size()) { outputs_dtypes_.resize(pos + 1, kFloat32); }
    outputs_dtypes_[pos] = dtype;
    return *static_cast<DerivedT*>(this);
  }

  [[nodiscard]] inline int getThreads() const { return threads_; }

  inline void setThreads(int threads) { threads_ = threads; }

  inline void setInputsConstant(size_t pos, uint8_t constant) {
    if (pos >= inputs_constant_.size()) { inputs_constant_.resize(pos + 1, 0); }
    inputs_constant_[pos] = constant;
  }

  inline void setOutputsConstant(size_t pos, uint8_t constant) {
    if (pos >= outputs_constant_.size()) { outputs_constant_.resize(pos + 1, 0); }
    outputs_constant_[pos] = constant;
  }

  [[nodiscard]] inline uint8_t getInputsConstant(size_t pos) const {
    if (pos >= inputs_constant_.size()) { return 0; }
    return inputs_constant_[pos];
  }

  [[nodiscard]] inline uint8_t getOutputsConstant(size_t pos) const {
    if (pos >= outputs_constant_.size()) { return 0; }
    return outputs_constant_[pos];
  }

  inline void setInplace(bool inplace) { inplace_ = inplace; }

  [[nodiscard]] bool isInplace() const { return inplace_; }

 private:
  bool inplace_ = false;
  int threads_ = 4;
  std::vector<DataTypes> inputs_dtypes_;
  std::vector<DataTypes> outputs_dtypes_;
  std::vector<uint8_t> inputs_constant_;
  std::vector<uint8_t> outputs_constant_;
};

// Type Erase
#ifdef MLLM_ENABLE_PY_MLLM
class BaseOpOptionsBase {
 public:
  // Do not mark this explicit
  template<typename T>
  BaseOpOptionsBase(const T& cargo)  // NOLINT(google-explicit-constructor)
      : inner_(std::make_unique<Model<T>>(cargo)) {}

  // Do not mark this explicit
  template<typename T>
  BaseOpOptionsBase(T&& cargo)  // NOLINT(google-explicit-constructor)
      : inner_(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(cargo))) {}

  template<typename T>
  const T& as() const {
    // FIXME: Unsafe cast.
    return static_cast<const Model<T>&>(*inner_).data_;
    // if (auto p = dynamic_cast<const Model<T>*>(inner_.get())) { return p->data_; }
    // throw std::bad_cast();
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
  };

  template<typename T>
  struct Model : Concept {
    Model(const T& data) : data_(data) {}  // NOLINT(google-explicit-constructor)
    T data_;
  };

  std::unique_ptr<Concept> inner_;
};
#else
class BaseOpOptionsBase {
 public:
  // Do not mark this explicit
  template<typename T>
  BaseOpOptionsBase(const T& cargo)  // NOLINT(google-explicit-constructor)
      : inner_(std::make_unique<Model<T>>(cargo)) {}

  // Do not mark this explicit
  template<typename T>
  BaseOpOptionsBase(T&& cargo)  // NOLINT(google-explicit-constructor)
      : inner_(std::make_unique<Model<std::decay_t<T>>>(std::forward<T>(cargo))) {}

  template<typename T>
  const T& as() const {
    if (auto p = dynamic_cast<const Model<T>*>(inner_.get())) { return p->data_; }
    throw std::bad_cast();
  }

 private:
  struct Concept {
    virtual ~Concept() = default;
  };

  template<typename T>
  struct Model : Concept {
    Model(const T& data) : data_(data) {}  // NOLINT(google-explicit-constructor)
    T data_;
  };

  std::unique_ptr<Concept> inner_;
};
#endif

class BaseOp : public std::enable_shared_from_this<BaseOp> {
 public:
  using ptr_t = std::shared_ptr<BaseOp>;

  explicit BaseOp(OpTypes op_type);

  virtual void load(const ParameterFile::ptr_t& ploader) {};

  virtual void trace(void* trace_context, const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {};

  virtual void forward(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {}

  virtual void reshape(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {}

  virtual void setup(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);

  virtual ParameterFile::ptr_t getParams() { return ParameterFile::create(); }

  [[nodiscard]] std::string getName() const;

  void setName(const std::string& name);

  [[nodiscard]] DeviceTypes getDevice() const;

  void setDeviceType(DeviceTypes device_type);

  OpTypes getOpType() const;

 private:
  DeviceTypes device_type_;
  std::string name_;
  OpTypes op_type_;
};

struct BaseOpFactory {
  virtual ~BaseOpFactory() = default;
  virtual std::shared_ptr<BaseOp> create(const BaseOpOptionsBase& base_cargo) = 0;
  [[nodiscard]] virtual OpTypes opType() const = 0;
  inline void __forceSetType(int32_t type) { type_ = type; }

  int32_t type_;
};

template<OpTypes type, typename CargoT>
class TypedOpFactory : public BaseOpFactory {
 public:
  TypedOpFactory() { __forceSetType(static_cast<int32_t>(type)); }

  std::shared_ptr<BaseOp> create(const BaseOpOptionsBase& base_cargo) override {
    const auto& cargo = base_cargo.as<CargoT>();
    return createOpImpl(cargo);
  }

  [[nodiscard]] OpTypes opType() const override { return (OpTypes)type_; }

 protected:
  virtual std::shared_ptr<BaseOp> createOpImpl(const CargoT& cargo) = 0;
};

}  // namespace mllm
