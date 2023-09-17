#ifndef MLLM_OP_H
#define MLLM_OP_H

#include "Tensor.hpp"
#include "Types.hpp"
#include <functional>
using std::function;
namespace mllm {

class Backend;
class Tensor;

class Op {
public:
  /**
   * @brief initializer.
   * @param backend   backend that exection will running on.
   */
  Op(){};
  // Op(const BackendType betype): backend_type_(betype) {};
  // Op() = delete;
  // Op(shared_ptr<Backend> bn) : backend_(bn) {
  //     // nothing to do
  // }
  Op(Backend *bn) : backend_(bn) {
    // nothing to do
  }
  virtual ~Op() = default;

  /**
   * @brief response shape change of input or output tensors.
   * 设定输入输出的tensor(已经to_cpu)
   * @param inputs    input tensors
   * @param outputs   output tensors
   * @return resize result
   */
  virtual ErrorCode Setup(vector<shared_ptr<Tensor>> &inputs,
                          vector<shared_ptr<Tensor>> &outputs) {
    // check inputs shape
    // Reshape outputs
    // Weight malloc set
    return NO_ERROR;
  }

  /**
   * @brief perform execution.
   * @param inputs    input tensors
   * @param outputs   output tensors
   * @return execution result
   */
  virtual ErrorCode Execute(vector<shared_ptr<Tensor>> &inputs,
                            vector<shared_ptr<Tensor>> &outputs) = 0;
  /**
   * @brief get backend.
   * @return backend.
   */
  // shared_ptr<Backend> backend() const {
  //     return backend_;
  // }
  Backend *backend() const { return backend_; }

private:
  // shared_ptr<Backend> backend_;
  Backend *backend_;
  vector<Tensor *> inputs_;
  vector<Tensor *> outputs_;
  // BackendType backend_type_;
  // tensor w
  // vector<>
};

// unordered_map<OpType, function<shared_ptr<Op>(Backend*)>> opMap;
} // namespace mllm

#endif // MLLM_OP_H