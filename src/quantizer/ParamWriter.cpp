//
// Created by lx on 23-10-30.
//

#include "ParamWriter.hpp"

ParamWriter::ParamWriter(std::string filename) :
    path_(std::move(filename)) {
    fp_ = fopen(path_.c_str(), "wb");
    writeInt(fp_, _MAGIC_NUMBER);
}
ParamWriter::~ParamWriter() {
    if (fp_ != nullptr)
        fclose(fp_);
}
int ParamWriter::calcIndexSize(const vector<string> names) {
    int size = 0;
    for (const auto &name : names) {
        // One Tensor Index Item Contains: Name_Len(Int)+Name(str)+Weights_Len(UInt64)+Offset(UInt64)+DataType(Int)
        size += sizeof(int) + name.size() + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(int);
    }
    return size;
}
void ParamWriter::writeIndex() {
    fseek(fp_, sizeof(int32_t) + sizeof(uint64_t), SEEK_SET);
    for (const auto &param : param_info_) {
        writeString(fp_, param.name);
        write_u64(fp_, param.size);
        write_u64(fp_, param.offset);
        writeInt(fp_, param.type);
    }
}

void ParamWriter::writeParam(string name, DataType type, void *data, uint64_t size) {
    auto &param = param_info_[index_];
    param.name = std::move(name);
    param.type = type;
    param.offset = ftell(fp_);
    fwrite(data, sizeof(char), size, fp_);
    param.size = ftell(fp_) - param.offset;
    index_++;
}
void ParamWriter::paddingIndex(const vector<string> names) {
    param_info_.resize(names.size());
    // write 0 padding to preserve space for index
    int index_size = calcIndexSize(names);
    write_u64(fp_, index_size);
    char i[index_size];
    fwrite(&i, sizeof(char), index_size, fp_);
}
