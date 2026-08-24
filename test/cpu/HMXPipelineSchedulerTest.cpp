#include "backends/cpu/AttentionPipelineExecutor.hpp"
#include "backends/cpu/HMXPipelineScheduler.hpp"
#include "backends/cpu/PipelineTaskExecutor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

using mllm::hmx_pipeline::FusedGroup;
using mllm::hmx_pipeline::HeadJob;
using mllm::hmx_pipeline::StreamingHeadGreedyScheduler;
using mllm::hmx_pipeline::StreamingHeadJob;
using mllm::hmx_pipeline::TwoLevelGreedyScheduler;

TEST(HMXPipelineSchedulerTest, InnerGreedyOrdersHeadsByPredictedMakespan) {
    const std::vector<FusedGroup> groups = {
        {7, 10.0, {{0, 1.0, 10.0}, {1, 8.0, 1.0}}},
    };
    const auto schedule = TwoLevelGreedyScheduler::plan(groups);
    ASSERT_EQ(schedule.groups.size(), 1U);
    EXPECT_EQ(schedule.groups[0].id, 7U);
    EXPECT_EQ(schedule.groups[0].head_order,
              (std::vector<std::size_t>{1, 0}));
    EXPECT_DOUBLE_EQ(schedule.sparse_finish_us, 29.0);
}

TEST(HMXPipelineSchedulerTest, OuterGreedyIsDeterministicAndNoWorseThanFifo) {
    const std::vector<FusedGroup> groups = {
        {3, 5.0, {{0, 7.0, 2.0}, {1, 1.0, 9.0}}},
        {1, 2.0, {{2, 2.0, 2.0}}},
        {2, 3.0, {{3, 5.0, 1.0}, {4, 1.0, 6.0}}},
    };
    const auto first = TwoLevelGreedyScheduler::plan(groups);
    const auto second = TwoLevelGreedyScheduler::plan(groups);
    const auto fifo = TwoLevelGreedyScheduler::fifo(groups);
    ASSERT_EQ(first.groups.size(), groups.size());
    EXPECT_EQ(first.sparse_finish_us, second.sparse_finish_us);
    for (std::size_t index = 0; index < first.groups.size(); ++index) {
        EXPECT_EQ(first.groups[index].id, second.groups[index].id);
        EXPECT_EQ(first.groups[index].head_order,
                  second.groups[index].head_order);
    }
    EXPECT_LE(first.sparse_finish_us, fifo.sparse_finish_us);
}

TEST(HMXPipelineSchedulerTest, FallsBackToFifoWhenGreedyCandidateIsWorse) {
    const std::vector<FusedGroup> groups = {
        {0, 5.0, {{0, 7.0, 10.0}, {1, 7.0, 10.0}}},
        {1, 15.0, {{2, 3.0, 10.0}}},
        {2, 6.0, {{3, 9.0, 8.0}}},
    };
    bool used_greedy = true;
    double candidate_greedy_us = 0.0;
    const auto selected = TwoLevelGreedyScheduler::planNoWorseThanFifo(
        groups, &used_greedy, &candidate_greedy_us);
    const auto fifo = TwoLevelGreedyScheduler::fifo(groups);
    EXPECT_FALSE(used_greedy);
    EXPECT_DOUBLE_EQ(candidate_greedy_us, 54.0);
    EXPECT_DOUBLE_EQ(fifo.sparse_finish_us, 50.0);
    EXPECT_DOUBLE_EQ(selected.sparse_finish_us, fifo.sparse_finish_us);
    ASSERT_EQ(selected.groups.size(), groups.size());
    for (std::size_t index = 0; index < selected.groups.size(); ++index) {
        EXPECT_EQ(selected.groups[index].id, fifo.groups[index].id);
        EXPECT_EQ(selected.groups[index].head_order,
                  fifo.groups[index].head_order);
    }
}

TEST(HMXPipelineSchedulerTest,
     StreamingHeadScheduleModelsPerHeadNpuPublication) {
    const std::vector<StreamingHeadJob> heads = {
        {4, 5.0, 2.0, 3.0},
        {7, 7.0, 2.0, 3.0},
    };
    const auto schedule = StreamingHeadGreedyScheduler::fifo(heads);
    EXPECT_EQ(schedule.order, (std::vector<std::size_t>{4, 7}));
    EXPECT_DOUBLE_EQ(schedule.npu_finish_us, 12.0);
    EXPECT_DOUBLE_EQ(schedule.topk_finish_us, 14.0);
    EXPECT_DOUBLE_EQ(schedule.sparse_finish_us, 17.0);
}

TEST(HMXPipelineSchedulerTest,
     StreamingHeadGreedyIsDeterministicAndNoWorseThanFifo) {
    const std::vector<StreamingHeadJob> heads = {
        {9, 1.0, 1.0, 10.0},
        {3, 1.0, 8.0, 1.0},
        {5, 2.0, 3.0, 4.0},
    };
    bool first_used_greedy = false;
    bool second_used_greedy = false;
    const auto first = StreamingHeadGreedyScheduler::planNoWorseThanFifo(
        heads, &first_used_greedy);
    const auto second = StreamingHeadGreedyScheduler::planNoWorseThanFifo(
        heads, &second_used_greedy);
    const auto fifo = StreamingHeadGreedyScheduler::fifo(heads);
    EXPECT_EQ(first.order, second.order);
    EXPECT_EQ(first_used_greedy, second_used_greedy);
    EXPECT_LE(first.sparse_finish_us, fifo.sparse_finish_us);
}

TEST(HMXPipelineSchedulerTest, CooperativeExecutorUsesAllWorkersPerSerialJob) {
    mllm::CooperativeTaskExecutor executor(2, {-1, -1}, "test Top-k");
    std::atomic<int> first_finished{0};
    std::atomic<bool> second_started_too_early{false};
    std::mutex lanes_mutex;
    std::set<std::size_t> first_lanes;
    std::set<std::size_t> second_lanes;
    auto first = executor.submit([&](std::size_t lane, std::size_t count) {
        EXPECT_EQ(count, 2U);
        {
            std::lock_guard<std::mutex> lock(lanes_mutex);
            first_lanes.insert(lane);
        }
        if (lane == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        ++first_finished;
    });
    auto second = executor.submit([&](std::size_t lane, std::size_t count) {
        EXPECT_EQ(count, 2U);
        if (first_finished.load() != 2) second_started_too_early = true;
        std::lock_guard<std::mutex> lock(lanes_mutex);
        second_lanes.insert(lane);
    });
    EXPECT_NO_THROW(first.get());
    EXPECT_NO_THROW(second.get());
    EXPECT_FALSE(second_started_too_early.load());
    EXPECT_EQ(first_lanes, (std::set<std::size_t>{0, 1}));
    EXPECT_EQ(second_lanes, (std::set<std::size_t>{0, 1}));
}

TEST(HMXPipelineSchedulerTest, PipelineExecutorReusesPersistentWorker) {
    mllm::PipelineTaskExecutor executor(nullptr, nullptr, "test stage", 1);
    std::thread::id first_worker;
    std::thread::id second_worker;
    auto first = executor.submit(
        [&]() { first_worker = std::this_thread::get_id(); });
    auto second = executor.submit(
        [&]() { second_worker = std::this_thread::get_id(); });
    ASSERT_NO_THROW(first.get());
    ASSERT_NO_THROW(second.get());
    EXPECT_NE(first_worker, std::this_thread::get_id());
    EXPECT_EQ(first_worker, second_worker);
}

TEST(HMXPipelineSchedulerTest, PipelineExecutorAcceptsMicrosecondSpinBudget) {
    constexpr const char *spin_env = "MLLM_TEST_PIPELINE_SPIN_US";
    ASSERT_EQ(setenv(spin_env, "20000", 1), 0);
    {
        mllm::PipelineTaskExecutor executor(
            nullptr, nullptr, "test stage", 1, nullptr, nullptr, spin_env);
        auto task = executor.submit([]() {});
        EXPECT_NO_THROW(task.get());
    }
    EXPECT_EQ(unsetenv(spin_env), 0);
}

TEST(HMXPipelineSchedulerTest, ParsesTypedLatencyProfileAndRejectsRetentionMismatch) {
    const std::string path = "/tmp/mllm-hmx-pipeline-profile-test.tsv";
    {
        std::ofstream output(path);
        ASSERT_TRUE(output.good());
        output << "#schema_version\t1\n"
               << "#model\tqwen2_1p5b\n"
               << "#device\thouji\n"
               << "#query_len\t128\n"
               << "#head_dim\t128\n"
               << "#max_key_len\t4160\n"
               << "#main_cpu\t5\n"
               << "#topk_cpus\t0,1\n"
               << "#sparse_cpus\t5\n"
               << "#manifest_sha256\tmanifest\n"
               << "#head_profile_sha256\tprofile\n"
               << "#binary_sha256\tbinary\n"
               << "#minimum_samples\t20\n"
               << "#statistic\tp50\n"
               << "#npu_key_cache_policy\tincremental\n"
               << "#npu_execution_scope\tattention\n"
               << "npu\t128\t2\t0.1\t0.2\t100\t120\n"
               << "topk\t128\t2\t3\t0.2\t30\t40\n"
               << "sparse\t128\t2\t3\t0.2\t50\t70\n";
    }
    const auto profile = mllm::hmx_pipeline::LatencyProfile::load(path);
    EXPECT_DOUBLE_EQ(profile.npu(128, 2, 0.1F, 0.2F).p50_us, 100.0);
    EXPECT_DOUBLE_EQ(profile.topk(128, 2, 3, 0.2F).p50_us, 30.0);
    EXPECT_DOUBLE_EQ(profile.sparse(128, 2, 3, 0.2F).p50_us, 50.0);
    EXPECT_NO_THROW(profile.validateRuntimeBinding(
        "qwen2_1p5b", "houji", 128, 128, 128, "5", "0,1", "5"));
    EXPECT_THROW(profile.validateRuntimeBinding(
        "qwen2_1p5b", "houji", 128, 128, 128, "5", "0", "5"),
        std::invalid_argument);
    EXPECT_THROW(profile.validateRuntimeBinding(
        "qwen2_1p5b", "houji", 128, 128, 4224, "5", "0,1", "5"),
        std::invalid_argument);
    EXPECT_THROW(profile.validateRuntimeBinding(
        "qwen2_1p5b", "houji", 128, 128, 128, "6", "0,1", "5"),
        std::invalid_argument);
    EXPECT_THROW(profile.validateRuntimeBinding(
        "qwen2_1p5b", "other", 128, 128, 128, "5", "0,1", "5"),
        std::invalid_argument);
    EXPECT_THROW(profile.topk(128, 2, 3, 0.3F), std::invalid_argument);
    (void)std::remove(path.c_str());
}

TEST(HMXPipelineSchedulerTest, RejectsDuplicateAndMissingProfileRows) {
    const std::string duplicate =
        "/tmp/mllm-hmx-pipeline-profile-duplicate.tsv";
    const std::string missing =
        "/tmp/mllm-hmx-pipeline-profile-missing.tsv";
    const std::string metadata =
        "#schema_version\t1\n"
        "#model\tqwen2_1p5b\n"
        "#device\thouji\n"
        "#query_len\t128\n"
        "#head_dim\t128\n"
        "#max_key_len\t4160\n"
        "#main_cpu\t5\n"
        "#topk_cpus\t0,1\n"
        "#sparse_cpus\t5\n"
        "#manifest_sha256\tmanifest\n"
        "#head_profile_sha256\tprofile\n"
        "#binary_sha256\tbinary\n"
        "#minimum_samples\t20\n"
        "#statistic\tp50\n"
        "#npu_key_cache_policy\tincremental\n"
        "#npu_execution_scope\tattention\n";
    {
        std::ofstream output(duplicate);
        output << metadata
               << "npu\t128\t2\t0.1\t0.2\t100\t120\n"
               << "npu\t128\t2\t0.1\t0.2\t101\t121\n"
               << "topk\t128\t2\t3\t0.2\t30\t40\n"
               << "sparse\t128\t2\t3\t0.2\t50\t70\n";
    }
    {
        std::ofstream output(missing);
        output << metadata
               << "npu\t128\t2\t0.1\t0.2\t100\t120\n"
               << "topk\t128\t2\t3\t0.2\t30\t40\n";
    }
    EXPECT_THROW(mllm::hmx_pipeline::LatencyProfile::load(duplicate),
                 std::invalid_argument);
    EXPECT_THROW(mllm::hmx_pipeline::LatencyProfile::load(missing),
                 std::invalid_argument);
    (void)std::remove(duplicate.c_str());
    (void)std::remove(missing.c_str());
}

} // namespace
