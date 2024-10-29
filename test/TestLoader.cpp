//
// Created by Xiang Li on 23-10-14.
//

#include "TestLoader.hpp"
#include <filesystem>

namespace mllm {
TestLoader::~TestLoader() {
}
TestLoader::TestLoader(string filename) :
    TestIO(filename, true) {
    if (fp_ == nullptr) {
        std::cout << filename << "File not found" << std::endl;
        return;
    }
    int magic = read_int();
    if (magic != 2233) {
        std::cout << "Magic number mismatch" << std::endl;
        fclose(fp_);
        fp_ = nullptr;
        return;
    }
    fseek(fp_, 0, SEEK_END);
    uint64_t end = ftell(fp_);
    fseek(fp_, 4, SEEK_SET);
    bool found_conv = (filename.find("CPUConvolution2D") != std::string::npos);
    bool found_conv3d = (filename.find("CPUConvolution3D") != std::string::npos);
    bool found_avgpool = (filename.find("CPUAvgPool2D") != std::string::npos);
    bool found_maxpool = (filename.find("CPUMaxPool2D") != std::string::npos);
    bool found = found_conv || found_avgpool || found_maxpool;
    while (ftell(fp_) != end) {
        string name = read_string();
        int type = read_int();
        vector<int> shape;
        if (found_conv3d)
            shape = read_shape(5);
        else
            shape = read_shape(4);
        uint64_t length = read_u64();
        TensorIndex *index = new TensorIndex();
        index->name = name;
        index->type = type;
        index->dims = shape;
        if (found & name == "input0") {
            index->dims = {shape[0], shape[2], shape[1], shape[3]};
        } else if (found & name == "output") {
            index->dims = {shape[0], shape[2], shape[1], shape[3]};
        }
        index->len = length;
        index->offset = ftell(fp_);
        tensor_map_[name] = index;
        fseek(fp_, length, SEEK_CUR);
    }
}
bool TestLoader::load(Tensor *tensor, bool strict) {
    if (fp_ == nullptr) {
        return false;
    }
    auto name = tensor->name();
    TensorIndex *index = tensor_map_[name];
    if (index == nullptr) {
        // Replace  all '.' with '_' in name
        string name_ = name;
        std::replace(name_.begin(), name_.end(), '.', '_');
        index = tensor_map_[name_];
        if (index == nullptr) {
            std::cout << "Tensor " << name << " not found" << std::endl;
            return false;
        }
    }
    if (index->dims.size() == 5) {
        tensor->chls() = {{BATCH, 0}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
        tensor->setCtype(BCTHW);
    }
    if (tensor->shape().empty()) {
        // Get shape from TensorIndex
        if (index->dims.size() == 5) {
            tensor->chls() = {{BATCH, 0}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
            tensor->reshape(index->dims[0], index->dims[1], index->dims[2], index->dims[3], index->dims[4]);
        } else {
            tensor->reshape(index->dims[0], index->dims[1], index->dims[2], index->dims[3]);
        }
        if (!tensor->allocted()) {
            tensor->alloc();
        }
    } else {
        if (!tensor->allocted()) {
            tensor->alloc();
        }
    }

    if (index->dims.size() == 5) {
        if ((!index->checkDim5({tensor->batch(), tensor->channel(), tensor->time(), tensor->height(), tensor->width()}, strict))) {
            return false;
        }
    } else {
        if ((!index->checkDim({tensor->batch(), tensor->head(), tensor->sequence(), tensor->dimension()}, strict))) {
            return false;
        }
    }

    fseek(fp_, index->offset, SEEK_SET);

    fread((void *)tensor->hostPtr<char>(), sizeof(uint8_t), index->len, fp_);
    return true;
}
bool TestLoader::load(shared_ptr<Tensor> tensor, bool strict) {
    return load(tensor.get(), strict);
}
DataType TestLoader::getDataType(string name) {
    //    return MLLM_TYPE_Q4_1;
    TensorIndex *index = tensor_map_[name];
    if (index == nullptr) {
        // Replace  all '.' with '_' in name
        string name_ = name;
        std::replace(name_.begin(), name_.end(), '.', '_');
        index = tensor_map_[name_];
        if (index == nullptr) {
            std::cout << "Tensor " << name << " not found" << std::endl;
            return DataType::MLLM_TYPE_COUNT;
        }
    }
    return static_cast<DataType>(index->type);
}
bool TestLoader::load(Tensor *tensor) {
    return load(tensor, false);
}
bool TestLoader::load(shared_ptr<Tensor> tensor) {
    return load(tensor, true);
}
uint64_t TestIO::read_u64() {
    uint64_t ret;
    fread(&ret, sizeof(uint64_t), 1, fp_);
    return ret;
}
int TestIO::read_int() {
    int ret;
    fread(&ret, sizeof(int), 1, fp_);
    return ret;
}
string TestIO::read_string() {
    int len = read_int();
    //    std::cout << "len:" << len << std::endl;

    char *buf = new char[len + 1];
    fread(buf, sizeof(char), len, fp_);
    buf[len] = '\0';
    //    std::cout << "buf:" << buf << std::endl;
    string ret(buf);
    delete[] buf;
    return ret;
}
double TestIO::read_f32() {
    float ret;
    fread(&ret, sizeof(float), 1, fp_);
    return ret;
}
vector<int> TestIO::read_shape() {
    int len = 4;
    vector<int> ret(len);
    for (int i = 0; i < len; ++i) {
        ret[i] = read_int();
        if (ret[i] < 0) {
            ret[i] = 1;
        }
    }
    return ret;
}
vector<int> TestIO::read_shape(int len) {
    vector<int> ret(len);
    for (int i = 0; i < len; ++i) {
        ret[i] = read_int();
        if (ret[i] < 0) {
            ret[i] = 1;
        }
    }
    return ret;
}
bool TestIO::write_string(string str) {
    if (read_mode_) return false;
    int len = str.length();
    int ret = -1;
    ret = fwrite(&len, sizeof(int), 1, fp_);
    if (ret < 0) {
        return false;
    }
    ret = fwrite(str.c_str(), sizeof(char), len, fp_);
    return ret >= 0;
}
bool TestIO::write_shape(vector<int> shape) {
    if (read_mode_) return false;
    int len = shape.size();
    for (int i = 0; i < len; ++i) {
        int ret = fwrite(&shape[i], sizeof(int), 1, fp_);
        if (ret < 0) {
            return false;
        }
    }
    return true;
}
bool TestIO::write_u64(uint64_t val) {
    if (read_mode_) return false;

    int ret = fwrite(&val, sizeof(uint64_t), 1, fp_);
    return ret >= 0;
}
bool TestIO::write_int(int val) {
    if (read_mode_) return false;
    return fwrite(&val, sizeof(int), 1, fp_) >= 0;
}

bool TensorIndex::checkDim(vector<int> dims_, bool strict) {
    if (dims_.size() != this->dims.size()) {
        std::cout << "dims size not match at " << this->name << " Expected: " << DimDesc(this->dims) << " Actual: " << DimDesc(dims_) << std::endl;
        return false;
    }
    if (!strict) {
        auto a_dim_num = dims[0] * dims[1] * dims[2] * dims[3];
        auto b_dim_num = dims_[0] * dims_[1] * dims_[2] * dims_[3];
        if (a_dim_num != b_dim_num) {
            std::cout << "dim num not match at " << this->name << " Expected: " << DimDesc(this->dims) << " Actual: " << DimDesc(dims_) << std::endl;
            return false;
        }
        return true;
    }
    for (int i = 0; i < 4; ++i) {
        if (dims_[i] != this->dims[i]) {
            std::cout << "dims not match at " << this->name << " Expected: " << DimDesc(this->dims) << " Actual: " << DimDesc(dims_) << std::endl;
            return false;
        }
    }
    return true;
}
bool TensorIndex::checkDim5(vector<int> dims_, bool strict) {
    if (dims_.size() != this->dims.size()) {
        std::cout << "dims size not match at " << this->name << " Expected: " << DimDesc(this->dims) << " Actual: " << DimDesc5(dims_) << std::endl;
        return false;
    }
    if (!strict) {
        auto a_dim_num = dims[0] * dims[1] * dims[2] * dims[3] * dims[4];
        auto b_dim_num = dims_[0] * dims_[1] * dims_[2] * dims_[3] * dims[4];
        if (a_dim_num != b_dim_num) {
            std::cout << "dim num not match at " << this->name << " Expected: " << DimDesc(this->dims) << " Actual: " << DimDesc5(dims_) << std::endl;
            return false;
        }
        return true;
    }
    for (int i = 0; i < 5; ++i) {
        if (dims_[i] != this->dims[i]) {
            std::cout << "dims not match at " << this->name << " Expected: " << DimDesc(this->dims) << " Actual: " << DimDesc(dims_) << std::endl;
            return false;
        }
    }
    return true;
}
string DimDesc(vector<int> dim) {
    dim.resize(4, 1);
    ostringstream ss;
    ss << "[" << dim[0] << "," << dim[1] << "," << dim[2] << "," << dim[3] << "]";
    return ss.str();
}
string DimDesc5(vector<int> dim) {
    dim.resize(5, 1);
    ostringstream ss;
    ss << "[" << dim[0] << "," << dim[1] << "," << dim[2] << "," << dim[3] << "," << dim[4] << "]";
    return ss.str();
}
TestIO::TestIO(string filename, bool read_mode) :
    read_mode_(read_mode) {
    filename = "test_" + filename + ".mllm";
    std::filesystem::path path = "../bin";
    path /= filename;
    filename = path.string();

    if (read_mode) {
        fp_ = fopen(filename.c_str(), "rb");
    } else {
        fp_ = fopen(filename.c_str(), "wb");
    }
}
TestIO::~TestIO() {
    if (fp_ != nullptr) {
        fclose(fp_);
    }
}
[[maybe_unused]] bool TestWriter::write(Tensor *tensor) {
    if (fp_ == nullptr || read_mode_ || tensor == nullptr) {
        return false;
    }
    if (!write_string(tensor->name())) {
        return false;
    }
    if (!write_int(-1)) {
        return false;
    }
    if (!write_shape(tensor->shape())) {
        return false;
    }
    uint64_t length = tensor->count() * tensor->dtypeSize();
    if (!write_u64(length)) {
        return false;
    }
    return fwrite(tensor->hostPtr<char>(), sizeof(uint8_t), length, fp_) > 0;
}
TestWriter::~TestWriter() {
}
[[maybe_unused]] TestWriter::TestWriter(string filename) :
    TestIO(filename, false) {
    if (fp_ == nullptr) {
        return;
    }
    write_int(2233);
}

} // namespace mllm