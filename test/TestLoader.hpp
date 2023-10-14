//
// Created by lx on 23-10-14.
//

#ifndef MLLM_TESTLOADER_HPP
#define MLLM_TESTLOADER_HPP
#include "string"
#include "Op.hpp"
#include "gtest/gtest.h"
using std::string;
namespace mllm {
// For serialize and deserialize Tensor in test.
// Separate from ParamLoader.
/*
 *  Format:
 *  Magic Number: Int
 *  Index length: u64
 *  -- Tensor 1 --
 *  Name: String(Len:Int + Char[])
 *  DataType: Int
 *  Dims: [n,c,h,w]
 *  Len: u64
 *  data: bytes
 */
struct TensorIndex {
    string name;
    int type;
    vector<int> dims;
    uint64_t len;
    uint64_t offset;
    bool checkDim(vector<int> dims_);
};
static string DimDesc(vector<int> dim);

class TestIO {
protected:
    explicit TestIO(string filename, bool read_mode);
    bool read_mode_;
    FILE *fp_;
    uint64_t read_u64();
    int read_int();
    string read_string();
    double read_f32();
    vector<int> read_shape();
    bool write_int(int val);
    bool write_u64(uint64_t val);
    bool write_shape(vector<int> shape);
    bool write_string(string str);
    ~TestIO();
};
class TestLoader : public TestIO {
public:
    explicit TestLoader(string filename);
    ~TestLoader();
    bool load(Tensor *tensor);
    bool load(shared_ptr<Tensor> tensor);

private:
    unordered_map<string, TensorIndex *> tensor_map_;
};
class TestWriter : public TestIO {
public:
    explicit TestWriter(string filename);
    ~TestWriter();
    bool Write(Tensor *tensor);
};

} // namespace mllm

#endif // MLLM_TESTLOADER_HPP
