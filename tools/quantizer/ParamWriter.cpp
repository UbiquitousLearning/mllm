//
// Created by Xiang Li on 23-10-30.
//
#include "ParamWriter.hpp"
#include <cstdio>
#include <utility>
#include <vector>
#include <string>

ParamWriter::ParamWriter(std::string filename) :
    path_(std::move(filename)) {
    fp_ = fopen(path_.c_str(), "wb");
    if (fp_ == nullptr) {
        throw std::runtime_error("Failed to open file for writing: " + path_);
    }
    // _MAGIC_NUMBER is defined in ParamLoader.hpp
    writeInt(fp_, _MAGIC_NUMBER);
}

ParamWriter::~ParamWriter() {
    if (fp_ != nullptr)
        fclose(fp_);
}

int ParamWriter::calcIndexSize(const std::vector<std::string> &names) {
    int size = 0;
    for (const auto &name : names) {
        // One Tensor Index Item Contains: Name_Len(Int)+Name(str)+Weights_Len(UInt64)+Offset(UInt64)+DataType(Int)
        size += sizeof(int32_t) + name.size() + sizeof(uint64_t) + sizeof(uint64_t) + sizeof(int32_t);
    }
    return size;
}

void ParamWriter::writeIndex() {
    fseek(fp_, sizeof(int32_t) + sizeof(uint64_t), SEEK_SET);
    for (const auto &param : param_info_) {
        writeString(fp_, param.name);
        write_u64(fp_, param.size);
        write_u64(fp_, param.offset);
        write_dtype(fp_, param.type);
    }
    fflush(fp_);
}

void ParamWriter::beginWriteParam(const std::string &name, DataType type) {
    if (index_ >= param_info_.size()) {
        throw std::runtime_error("Parameter index out of bounds. Did you call paddingIndex correctly?");
    }
    auto &param = param_info_[index_];
    param.name = name;
    param.type = type;
    param.offset = ftell(fp_);

    current_param_start_offset_ = param.offset;
}

void ParamWriter::writeChunk(const void *data, uint64_t size_in_bytes) {
    if (size_in_bytes == 0) return;
    auto status = fwrite(data, 1, size_in_bytes, fp_);
    if (status != size_in_bytes) {
        std::cout << "fwrite error: wrote " << status << " bytes instead of " << size_in_bytes << std::endl;
        throw std::runtime_error("Failed to write chunk to file.");
    }
}

void ParamWriter::endWriteParam() {
    fflush(fp_);
    auto current_pos = ftell(fp_);
    if (index_ >= param_info_.size()) {
        throw std::runtime_error("Parameter index out of bounds at endWriteParam.");
    }
    auto &param = param_info_[index_];
    param.size = current_pos - current_param_start_offset_;

    index_++;
}

void ParamWriter::paddingIndex(const std::vector<std::string> &names) {
    param_info_.resize(names.size());
    int index_size = calcIndexSize(names);
    write_u64(fp_, index_size);
    std::vector<char> padding(index_size, 0);
    fwrite(padding.data(), sizeof(char), index_size, fp_);
}