#pragma once
// #include <algorithm>
// #include <map>
#include <iostream>
// #include <map>
// #include <ostream>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include "Tensor.hpp"

using namespace std;
using namespace mllm;

int mbp_load_layer_idx;
int mbp_load_expert_idx;

struct LoadRequest {
    int layer;
    int expert;
};
queue<LoadRequest> load_requests; // 替换原do_mbp_load相关变量
mutex queue_mutex;                // 队列互斥锁
condition_variable queue_cv;      // 队列条件变量

// ========= Clip Thread Globals (NEW) =========
// 1. 新增 ClipRequest 结构体
struct ClipRequest {
    int layer;
    int expert;
    Tensor hidden_states;
    Tensor exp_token_idx;
    Tensor topk_weight;
    Tensor exp_idx;
};
// 2. 新增 clip 线程的任务队列、锁和条件变量
queue<ClipRequest> clip_requests;
mutex clip_queue_mutex;
condition_variable clip_queue_cv;
// 3. 新增用于存储 clip 结果的 map 和其互斥锁
map<string, pair<Tensor, Tensor>> clipped_data;
mutex clip_results_mutex;
//============ End Clip Thread Globals ============

atomic<bool> mbp_finish{false}; // 改为原子布尔

vector<vector<unique_ptr<mutex>>> mtxs;             // 每个层和专家一个互斥锁
vector<vector<unique_ptr<condition_variable>>> cvs; // 每个层和专家一个条件变量
vector<vector<atomic<bool>>> dones;                 // 原子布尔保证可见性

// --- Clipping Primitives (NEW) ---
// 4. 新增 clip 线程的同步对象
vector<vector<unique_ptr<mutex>>> clip_mtxs;
vector<vector<unique_ptr<condition_variable>>> clip_cvs;
vector<vector<atomic<bool>>> clip_dones;

// 修改 MAP_MINICPMMOE_MBP_HPP 中的相关部分

inline void reset_syntax_mbm(int layer_idx, int expert_idx) {
    // 使用原子操作重置状态
    dones[layer_idx][expert_idx].store(false, std::memory_order_release);
}

inline void mbp_init(int num_layers, int num_experts) {
    // 初始化 loading 相关的变量
    mtxs.resize(num_layers);
    cvs.resize(num_layers);
    dones.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        mtxs[i].resize(num_experts);
        cvs[i].resize(num_experts);
        dones[i] = std::vector<std::atomic<bool>>(num_experts);
        for (int j = 0; j < num_experts; ++j) {
            mtxs[i][j] = make_unique<mutex>();
            cvs[i][j] = make_unique<condition_variable>();
            dones[i][j].store(false, std::memory_order_relaxed);
        }
    }
    // 初始化 clipping 相关的变量
    clip_mtxs.resize(num_layers);
    clip_cvs.resize(num_layers);
    clip_dones.resize(num_layers);
    for (int i = 0; i < num_layers; ++i) {
        clip_mtxs[i].resize(num_experts);
        clip_cvs[i].resize(num_experts);
        clip_dones[i] = std::vector<std::atomic<bool>>(num_experts);
        for (int j = 0; j < num_experts; ++j) {
            clip_mtxs[i][j] = make_unique<mutex>();
            clip_cvs[i][j] = make_unique<condition_variable>();
            clip_dones[i][j].store(false, std::memory_order_relaxed);
        }
    }
}

map<string, pair<double, double>> load_times;
map<string, pair<double, double>> expert_cal_times;
map<string, pair<double, double>> expert_clip_times;
map<string, pair<double, double>> expert_wait_times;
uint64_t start_time;
void clearMBPtimes() {
    load_times.clear();
    expert_cal_times.clear();
    expert_clip_times.clear();
    expert_wait_times.clear();
    clipped_data.clear();
    start_time = 0;
}
void prinMBPtimes(string start_word = "") {
    double load_times_cal = 0;
    cout << "load_times = [" << endl;
    for (const auto &entry : load_times) {
        if (start_word.empty() || entry.first.substr(0, start_word.length()) == start_word) {
            cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
        }
        load_times_cal += entry.second.second - entry.second.first;
    }
    cout << "]" << endl;
    cout << "calc_times = [" << endl;
    for (const auto &entry : expert_cal_times) {
        if (start_word.empty() || entry.first.substr(0, start_word.length()) == start_word) {
            cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
        }
    }
    cout << "]" << endl;
    cout << "clip_times = [" << endl;
    for (const auto &entry : expert_clip_times) {
        if (start_word.empty() || entry.first.substr(0, start_word.length()) == start_word) {
            cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
        }
    }
    cout << "]" << endl;
    cout << "wait_times = [" << endl;
    for (const auto &entry : expert_wait_times) {
        if (start_word.empty() || entry.first.substr(0, start_word.length()) == start_word) {
            cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
        }
    }
    cout << "]" << endl;
    std::cout << "load_times_cal = " << load_times_cal << "ms" << endl;
}