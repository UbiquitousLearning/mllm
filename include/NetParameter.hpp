
#ifndef MLLM_NETPARAMETER_H
#define MLLM_NETPARAMETER_H

#include "Types.hpp"
#include <algorithm>
#include<string.h>
#include <string>
#include <vector>
#include <iostream>
#include <string>  
#include <iostream> 
#include <map>
#include <sstream>
using std::vector;
using std::string;
using std::map;

namespace mllm {
    typedef struct{
        BackendType bntype;
        vector<string> op_names_;//{o1,o2,o3}
        vector<vector<string>> op_in_names_;//{{in}, {op1}. {in,op2}}
        // vector<optype>;
    }NetParameter;

    typedef map<std::string, int> OpParam;


}

#endif //MLLM_NETPARAMETER_H