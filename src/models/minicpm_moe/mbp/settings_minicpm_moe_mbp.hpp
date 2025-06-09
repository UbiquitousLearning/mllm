#ifndef MAP_MINICPMMOE_MBP_HPP
#define MAP_MINICPMMOE_MBP_HPP
// #include <algorithm>
// #include <map>
#include <iostream>
#include <map>
// #include <ostream>
#include <sys/types.h>
#include <utility>
#include <vector>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>

using namespace std;

int mbp_load_layer_idx;
int mbp_load_expert_idx;

struct LoadRequest {
    int layer;
    int expert;
};

queue<LoadRequest> load_requests; // 替换原do_mbp_load相关变量
mutex queue_mutex;                // 队列互斥锁
condition_variable queue_cv;      // 队列条件变量
atomic<bool> mbp_finish{false};   // 改为原子布尔

vector<vector<unique_ptr<mutex>>> mtxs;             // 每个层和专家一个互斥锁
vector<vector<unique_ptr<condition_variable>>> cvs; // 每个层和专家一个条件变量
vector<vector<atomic<bool>>> dones;                 // 原子布尔保证可见性

// 修改 MAP_MINICPMMOE_MBP_HPP 中的相关部分

inline void reset_syntax_mbm(int layer_idx, int expert_idx) {
    // 使用原子操作重置状态
    dones[layer_idx][expert_idx].store(false, std::memory_order_release);
}

inline void minicpmmoe_mbp_init(int num_layers, int num_experts) {
    mtxs.resize(num_layers);
    cvs.resize(num_layers);
    dones.resize(num_layers);

    for (int i = 0; i < num_layers; ++i) {
        mtxs[i].resize(num_experts);
        cvs[i].resize(num_experts);
        dones[i] = std::vector<std::atomic<bool>>(num_experts); // Step 2: 显式构造

        for (int j = 0; j < num_experts; ++j) {
            mtxs[i][j] = make_unique<mutex>();
            cvs[i][j] = make_unique<condition_variable>();
            dones[i][j].store(false, std::memory_order_relaxed); // 显式原子初始化
        }
    }
}

map<string, pair<double, double>> load_times;
map<string, pair<double, double>> expert_cal_times;
map<string, pair<double, double>> expert_clip_times;
uint64_t start_time;
void prinMBPtimes() {
    double load_times_cal = 0;
    cout << "load_times = [" << endl;
    for (const auto &entry : load_times) {
        cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
        load_times_cal += entry.second.second - entry.second.first;
    }
    cout << "]" << endl;
    cout << "calc_times = [" << endl;
    for (const auto &entry : expert_cal_times) {
        cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
    }
    cout << "]" << endl;
    cout << "clip_times = [" << endl;
    for (const auto &entry : expert_clip_times) {
        cout << "(\"" << entry.first << "\" , " << entry.second.first << ", " << entry.second.second << ")," << endl;
    }
    cout << "]" << endl;
    std::cout << "load_times_cal = " << load_times_cal << "ms" << endl;
}
#endif // MAP_MINICPMMOE_MBP_HPP