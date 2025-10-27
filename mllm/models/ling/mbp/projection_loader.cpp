#include "settings_bailing_moe_mbp_e.hpp"
#include "Timing.hpp"

using namespace std;
using namespace mllm;

// 投影层加载线程函数的实现
void projection_loading_thread_func() {
    while (!mbp_finish.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(projection_queue_mutex);
        projection_queue_cv.wait(lk, [] {
            return !projection_load_requests.empty() || mbp_finish.load(std::memory_order_acquire);
        });

        if (mbp_finish.load(std::memory_order_acquire)) {
            break;
        }

        while (!projection_load_requests.empty()) {
            auto req = projection_load_requests.top(); // 从优先队列的顶部取出最高优先级的请求
            projection_load_requests.pop();
            lk.unlock();

            // 执行投影层加载
            {
                std::unique_lock<std::mutex> proj_lk(*proj_mtxs[req.layer][req.expert][req.projection_type]);
                if (!proj_dones[req.layer][req.expert][req.projection_type].load(std::memory_order_acquire)) {
                    double time_start = (mllm_time_us() - start_time) / 1000.0F;

                    // 使用外部加载函数
                    if (load_projection_impl) {
                        load_projection_impl(req.layer, req.expert, req.projection_type);
                    }

                    proj_dones[req.layer][req.expert][req.projection_type].store(true, std::memory_order_release);

                    // 统一key命名格式：{layer}_{expert}_{proj_type}
                    string proj_type_name;
                    switch (req.projection_type) {
                    case 0: proj_type_name = "gate"; break;
                    case 1: proj_type_name = "up"; break;
                    case 2: proj_type_name = "down"; break;
                    default: proj_type_name = "unknown"; break;
                    }
                    string proj_name = std::to_string(req.layer) + "_" + std::to_string(req.expert) + "_" + proj_type_name;
                    double time_end = (mllm_time_us() - start_time) / 1000.0F;
                    proj_load_times[proj_name] = {time_start, time_end};
                    // std::cout << "Projection loaded: " << proj_name << std::endl;

                    // 刷新输出缓冲区，确保日志立即显示
                    std::cout.flush();
                }
            }
            proj_cvs[req.layer][req.expert][req.projection_type]->notify_all();
            lk.lock();
        }
    }
}

// clip线程函数的实现
void clip_thread_func() {
    while (!mbp_finish.load(std::memory_order_acquire)) {
        std::unique_lock<std::mutex> lk(clip_queue_mutex);
        clip_queue_cv.wait(lk, [] {
            return !clip_requests.empty() || mbp_finish.load(std::memory_order_acquire);
        });

        if (mbp_finish.load(std::memory_order_acquire)) {
            break;
        }

        while (!clip_requests.empty()) {
            auto req = clip_requests.front();
            clip_requests.pop();
            lk.unlock();

            // 执行clip操作
            {
                std::unique_lock<std::mutex> clip_lk(*clip_mtxs[req.layer][req.expert]);
                if (!clip_dones[req.layer][req.expert].load(std::memory_order_acquire)) {
                    double time_start = (mllm_time_us() - start_time) / 1000.0F;

                    // 执行实际的clip操作
                    auto expert_tokens = req.hidden_states.clip(req.exp_token_idx, SEQUENCE);
                    auto topk_weight_clip = req.topk_weight.clip(req.exp_idx, SEQUENCE);

                    // 存储clip结果
                    string key = std::to_string(req.layer) + "_" + std::to_string(req.expert);
                    {
                        std::lock_guard<std::mutex> results_lk(clip_results_mutex);
                        clipped_data[key] = {expert_tokens, topk_weight_clip};
                    }

                    clip_dones[req.layer][req.expert].store(true, std::memory_order_release);

                    double time_end = (mllm_time_us() - start_time) / 1000.0F;
                    expert_clip_times[key] = {time_start, time_end};
                }
            }
            clip_cvs[req.layer][req.expert]->notify_all();
            lk.lock();
        }
    }
}
