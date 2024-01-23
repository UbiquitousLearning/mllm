//
// Created by Xiang Li on 23-10-14.
//

#ifndef MLLM_TESTLOADER_HPP
#define MLLM_TESTLOADER_HPP
#include "ParamLoader.hpp"
#include "string"
#include "Op.hpp"
#include "gtest/gtest.h"
#include <unordered_map>
using std::string;
using namespace std;
namespace mllm {
// For serialize and deserialize Tensor in test.
// Separate from ParamLoader, since we do not use Index here.
/*
 *  Format:
 *  Magic Number: Int
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
    bool checkDim(vector<int> dims_, bool strict);
    bool checkDim5(vector<int> dims_, bool strict);
};
static string DimDesc(vector<int> dim);
static string DimDesc5(vector<int> dim);

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
    vector<int> read_shape(int len);
    bool write_int(int val);
    bool write_u64(uint64_t val);
    bool write_shape(vector<int> shape);
    bool write_string(string str);
    ~TestIO();
};
class TestLoader : public TestIO, public AbstructLoader {
public:
    explicit TestLoader(string filename);
    ~TestLoader();
    DataType getDataType(string name) override;
    bool load(Tensor *tensor) override;
    bool load(shared_ptr<Tensor> tensor) override;
    bool load(Tensor *tensor, bool strict);
    bool load(shared_ptr<Tensor> tensor, bool strict);

private:
    unordered_map<string, TensorIndex *> tensor_map_;
};
class TestWriter : public TestIO {
public:
    [[maybe_unused]] explicit TestWriter(string filename);
    ~TestWriter();
    [[maybe_unused]] bool write(Tensor *tensor);
};

} // namespace mllm

#endif // MLLM_TESTLOADER_HPP
