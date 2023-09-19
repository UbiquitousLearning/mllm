
#ifndef MLLM_NETPARAMETER_H
#define MLLM_NETPARAMETER_H

#include "Types.hpp"
#include <algorithm>
#include <iostream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string.h>
#include <string>
#include <vector>
#include "set"
using std::string;
using std::vector;
using std::map;

namespace mllm {

typedef map<std::string, int> OpParam;
typedef struct TNetTensor NetTensor;
typedef struct TNetParameter NetParameter;

typedef struct TNetOp {
    OpType type;
    vector<NetTensor *> in;
    vector<NetTensor *> out;
    vector<string> inOp; // input ops' names;
    string name;
    OpParam param;
    // ~TNetOp() {
    //     std::cout << "delete op: " << name << std::endl;
    // }
} NetOp;
typedef struct TNetTensor {
    string name;
    vector<int> shape;
    DataType type;
    NetOp *in;
    vector<NetOp *> out;
    NetParameter *subgraph;
} NetTensor;

typedef struct TNetParameter {
    string weights_path;
    string model_path;
    // string input_name;
    // string output_name;
    vector<NetOp *> net_ops;
    vector<NetTensor *> net_tensors;
    std::set<NetTensor *> net_inputs;
    std::set<NetTensor *> net_outputs;

} NetParameter;

} // namespace mllm

#endif // MLLM_NETPARAMETER_H