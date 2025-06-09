#ifndef MLLM_TENSORIMPL_H
#define MLLM_TENSORIMPL_H
#include <climits>
#include "Backend.hpp"
#include "OpDefined.hpp"
#include <iostream>
#include <cstdio>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <map>
#include <memory>
#include <vector>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif
#include <Types.hpp>
#include <assert.h>
// #include <sys/stat.h>

namespace mllm {
class Backend;
class Module;

class TensorImpl {
public:
    bool owns_host_ptr_ = true; // 新增标志位

    std::map<Chl, int> chls_ = {{BATCH, 0}, {SEQUENCE, 1}, {HEAD, 2}, {DIMENSION, 3}, {CHANNLE, 1}, {TIME, 2}, {HEIGHT, 3}, {WIDTH, 4}};
    string name_;
    DataType dtype_ = MLLM_TYPE_F32;
    ChlType ctype_ = BSHD;

    Backend *backend_ = nullptr;
    void *host_ptr_ = nullptr;

    vector<uint64_t> shape_;
    uint64_t capacity_ = 0;
    uint64_t count_ = 0;
    uint64_t allocated_ = 0;

    bool transed_ = false;
    bool should_in_graphs_ = true;

    bool undiffusion_ = false;
    vector<std::pair<Chl, Chl>> trans_from_;

    Module *module_ = nullptr;

    // 构造函数
    TensorImpl() = default;
    TensorImpl(Backend *bn) :
        backend_(bn) {
    }

    // 析构函数负责资源释放
    ~TensorImpl() {
        if (host_ptr_ != nullptr && owns_host_ptr_) {
            if (backend_) {
                backend_->free(host_ptr_);
            }
        }
    }

    // 禁止拷贝（使用shared_ptr管理）
    TensorImpl(const TensorImpl &) = delete;
    TensorImpl &operator=(const TensorImpl &) = delete;

    size_t cntSize() {
        return DataTypeSize(dtype_, count_);
    }
    void alloc() {
        if (allocated_ != count_) {
            if (host_ptr_ != nullptr && owns_host_ptr_) {
                backend_->free(host_ptr_);
                host_ptr_ = nullptr;
            }
            if (count_ > 0) {
                backend_->alloc(&host_ptr_, cntSize() + 16, 128);
            }
            allocated_ = count_;
        }
    }

    void free() {
        if (host_ptr_ != nullptr && owns_host_ptr_) { // 直接访问成员变量
            if (backend_) {
                backend_->free(host_ptr_);
            }
            host_ptr_ = nullptr;
            allocated_ = 0;
        }
    }

    //
    int canonicalAxisIndex(int axis_index) const {
        if (axis_index < 0) {
            return axis_index + shape_.size();
        }
        return axis_index;
    }
    int shape(int index) const {
        return shape_[canonicalAxisIndex(index)];
    }

    int numAxes() const {
        return shape_.size();
    }

    int legacyShape(int index) const {
        if (index >= numAxes() || index < -numAxes()) {
            return 0;
        }
        return shape(index);
    }

    std::map<Chl, int> &chls() {
        return chls_;
    }

    int batch() {
        return legacyShape(chls()[BATCH]);
    }
    int head() {
        return legacyShape(chls()[HEAD]);
    }
    int sequence() {
        return legacyShape(chls()[SEQUENCE]);
    }
    int dimension() {
        return legacyShape(chls()[DIMENSION]);
    }
    int channel() {
        assert(shape_.size() == 5);
        return legacyShape(chls()[CHANNLE]);
    }
    int time() {
        assert(shape_.size() == 5);
        return legacyShape(chls()[TIME]);
        switch (ctype_) {
        case BCTHW:
            return legacyShape(2);
        case BTHWC:
            return legacyShape(1);
        default: return -1;
        }
    }
    int height() {
        assert(shape_.size() == 5);
        return legacyShape(chls()[HEIGHT]);
    }
    int width() {
        assert(shape_.size() == 5);
        return legacyShape(chls()[WIDTH]);
    }

    bool private_reshape(const vector<int> &shape) {
        assert(shape.size() <= 32);
        count_ = 1;
        shape_.resize(shape.size());
        for (int i = 0; i < shape.size(); ++i) {
            assert(shape[i] >= 0);
            if (count_ != 0) {
                assert(shape[i] <= std::numeric_limits<uint64_t>::max() / count_);
            }
            count_ *= shape[i];
            shape_[i] = shape[i];
        }
        if (count_ > capacity_) {
            capacity_ = count_;
            return true;
        }
        return false;
    }

    bool reshape(const int batch, const int head, const int sequence, const int dimension) {
        vector<int> shape(4);
        shape[chls()[BATCH]] = batch;
        shape[chls()[HEAD]] = head;
        shape[chls()[SEQUENCE]] = sequence;
        shape[chls()[DIMENSION]] = dimension;
        return private_reshape(shape);
    }

    void changeCtype(int size = 0) {
        if (!shape_.empty()) {
            size = shape_.size();
        }
        // BCTHW = 3,
        // BTHWC = 4,
        // BWCTH = 5,
        if (size == 5 || (ctype_ >= 3 && ctype_ <= 5)) {
            vector<int> a = {chls()[BATCH], chls()[TIME], chls()[HEIGHT], chls()[WIDTH], chls()[CHANNLE]};
            ctype_ = Chls2Type[a];
        } else {
            vector<int> a = {chls()[BATCH], chls()[HEAD], chls()[SEQUENCE], chls()[DIMENSION]};
            ctype_ = Chls2Type[a];
        }
    }
};

} // namespace mllm

#endif // MLLM_TENSORIMPL_H