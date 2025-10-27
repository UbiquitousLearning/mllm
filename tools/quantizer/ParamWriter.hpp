//
// Created by Xiang Li on 23-10-30.
//

#ifndef MLLM_PARAMWRITER_HPP
#define MLLM_PARAMWRITER_HPP
#include "ParamLoader.hpp"
#include <vector>
#include <string>
#include <cstring>

static void write_u64(FILE *fp, uint64_t val) {
    fwrite(&val, sizeof(uint64_t), 1, fp);
}
static void writeInt(FILE *fp, int32_t val) {
    fwrite(&val, sizeof(int32_t), 1, fp);
}
static void writeString(FILE *fp, const std::string &str) {
    writeInt(fp, static_cast<int32_t>(str.size()));
    if (!str.empty()) {
        fwrite(str.c_str(), sizeof(char), str.size(), fp);
    }
}
static void write_dtype(FILE *fp, DataType dtype) {
    writeInt(fp, static_cast<int32_t>(dtype));
}

struct ParmInfo {
    std::string name;
    DataType type;
    uint64_t offset;
    uint64_t size;
};

class ParamWriter {
public:
    virtual ~ParamWriter();
    explicit ParamWriter(std::string filename);
    int calcIndexSize(const std::vector<std::string> &names);
    void writeIndex();

    void beginWriteParam(const std::string &name, DataType type);
    void writeChunk(const void *data, uint64_t size_in_bytes);
    void endWriteParam();

    void paddingIndex(const std::vector<std::string> &names);

protected:
    uint64_t index_ = 0;
    FILE *fp_;
    std::string path_;
    std::vector<ParmInfo> param_info_;

private:
    uint64_t current_param_start_offset_ = 0;
};

#endif // MLLM_PARAMWRITER_HPP