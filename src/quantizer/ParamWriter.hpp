//
// Created by Xiang Li on 23-10-30.
//

#ifndef MLLM_PARAMWRITER_HPP
#define MLLM_PARAMWRITER_HPP
#include "ParamLoader.hpp"
static void write_u64(FILE *fp, uint64_t val) {
    fwrite(&val, sizeof(uint64_t), 1, fp);
}
static void writeInt(FILE *fp, int32_t val) {
    fwrite(&val, sizeof(int32_t), 1, fp);
}
static void writeString(FILE *fp, const std::string &str) {
    writeInt(fp, str.size());
    fwrite(str.c_str(), sizeof(char), str.size(), fp);

}
static void write_dtype(FILE *fp, DataType dtype) {
    writeInt(fp, dtype);
}

struct ParmInfo {
    std::string name;
    DataType type;
    uint64_t offset;
    uint64_t size;
};
class ParamWriter {
public:
    ~ParamWriter();
    ParamWriter(std::string filename);
    int calcIndexSize(vector<string> names);
    void writeIndex();
    virtual void writeParam(string name, DataType type, void *data, uint64_t size);
    void paddingIndex(vector<string> names);

private:
    uint64_t index_ = 0;
    FILE *fp_;
    std::string path_;
    std::vector<ParmInfo> param_info_;
};

#endif // MLLM_PARAMWRITER_HPP
