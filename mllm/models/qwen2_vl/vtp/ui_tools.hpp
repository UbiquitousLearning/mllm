
#ifndef UI_TOOLS_HPP
#define UI_TOOLS_HPP

// 全局区域掩码
#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>
#include <algorithm> // 需要包含 <algorithm> 用于 std::sort 和 std::unique
#include <cstdint>   // 需要包含 <cstdint> 用于 uint32_t
bool use_pre_vit_merge = false;

std::vector<uint32_t> UIRegionMask;
// 输入类型为二维向量 [batch][patch_size]
std::vector<int> process_region_mask(const std::vector<std::vector<uint32_t>> &region_masks) {
    // 1. 验证批次大小是否为1
    const int batch_size = region_masks.size();
    if (batch_size != 1) {
        throw std::runtime_error("Batch size must be 1");
    }

    // 存储每个唯一标签选中的索引
    std::vector<int> selected_indices;

    // 随机数引擎
    std::random_device rd;
    std::mt19937 rng(rd());

    // 2. 处理批次 (循环只会执行一次)
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        const auto &region_mask = region_masks[batch_idx];
        const int patch_size = region_mask.size();

        // 3. 获取当前批次的唯一且已排序的标签 (这是关键的修改点)
        //---------------------------------------------------------
        // 旧的、基于 unordered_set 的错误方法:
        // std::unordered_set<uint32_t> unique_labels;
        // for (int i = 0; i < patch_size; ++i) {
        //     unique_labels.insert(region_mask[i]);
        // }
        //---------------------------------------------------------

        // 新的、正确的、模仿 torch.unique() 的方法:
        std::vector<uint32_t> unique_labels = region_mask;     // 复制一份
        std::sort(unique_labels.begin(), unique_labels.end()); // 排序
        // 移除相邻的重复元素，并调整 vector 大小
        unique_labels.erase(std::unique(unique_labels.begin(), unique_labels.end()), unique_labels.end());

        // 4. 为每个标签随机选择一个索引
        for (uint32_t label : unique_labels) {
            // 收集所有等于当前标签的索引位置
            std::vector<int> indices;
            for (int i = 0; i < patch_size; ++i) {
                if (region_mask[i] == label) {
                    indices.push_back(i);
                }
            }

            // 验证是否有区域存在 (理论上不会触发，因为标签来自 region_mask 本身)
            if (indices.empty()) {
                throw std::runtime_error("No region mask found for a label that should exist.");
            }

            // 随机选择一个索引
            std::uniform_int_distribution<int> dist(0, indices.size() - 1);
            int selected_idx = indices[dist(rng)];
            selected_indices.push_back(selected_idx);
        }
    }

    // 5. 扩展索引 (这部分逻辑你的实现是完全正确的)
    // 对应 PyTorch 的: selected_indices.unsqueeze(1) * 4 + torch.arange(4)
    // 和 .flatten()
    std::vector<int> final_indices;
    // 预分配内存以提高效率
    final_indices.reserve(selected_indices.size() * 4);

    for (int idx : selected_indices) {
        for (int ch = 0; ch < 4; ++ch) {
            final_indices.push_back(idx * 4 + ch);
        }
    }

    return final_indices;
}

#endif