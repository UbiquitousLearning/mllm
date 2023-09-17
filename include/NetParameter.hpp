
#ifndef MLLM_NETPARAMETER_H
#define MLLM_NETPARAMETER_H

#include "Types.hpp"
#include <algorithm>
#include <iostream>
#include <iostream>
#include <map>
#include <sstream>
#include <string.h>
#include <string>
#include <string>
#include <vector>
using std::string;
using std::vector;
using std::map;

namespace mllm {
typedef struct {
  struct NetOp {
    OpType type;
    vector<int> in;
    vector<int> out;
    string name;
  };
  struct NetTensor {
    string name;
    vector<int> shape;
    DataType type;
    vector<int> in;
    vector<int> out;
  };
  string weights_path;
  string model_path;
  string input_name;
  string output_name;
  vector<NetOp> net_ops;
  vector<NetTensor> net_tensors;

} NetParameter;

typedef map<std::string, int> OpParam;


} // namespace mllm

#endif // MLLM_NETPARAMETER_H