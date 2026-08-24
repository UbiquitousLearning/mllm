#include "Parallel.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

using namespace mllm;

TEST(ChunkPipelineTest, CountsExactMultiplesOnce) {
    EXPECT_EQ(ChunkPipeline::chunkCountFor(256, 256), 1);
    EXPECT_EQ(ChunkPipeline::chunkCountFor(1024, 256), 4);
    EXPECT_EQ(ChunkPipeline::chunkCountFor(4096, 256), 16);
    EXPECT_EQ(ChunkPipeline::paddedSequenceLengthFor(1024, 256), 1024);
}

TEST(ChunkPipelineTest, PadsPartialAndOddChunkCounts) {
    EXPECT_EQ(ChunkPipeline::chunkCountFor(257, 256), 2);
    EXPECT_EQ(ChunkPipeline::chunkCountFor(1025, 256), 5);
    EXPECT_EQ(ChunkPipeline::paddedSequenceLengthFor(1025, 256), 1280);

    ChunkPipeline pipeline(1025, 256);
    EXPECT_EQ(pipeline.chunkCount(), 5);
    EXPECT_EQ(pipeline.paddedSequenceLength(), 1280);
}

TEST(ChunkPipelineTest, SelectsLastRealTokenWithinFinalChunk) {
    EXPECT_EQ(ChunkPipeline::lastTokenIndexFor(1, 256), 0);
    EXPECT_EQ(ChunkPipeline::lastTokenIndexFor(256, 256), 255);
    EXPECT_EQ(ChunkPipeline::lastTokenIndexFor(257, 256), 0);
    EXPECT_EQ(ChunkPipeline::lastTokenIndexFor(1024, 256), 255);
}

TEST(ChunkPipelineTest, RejectsInvalidLengths) {
    EXPECT_THROW(ChunkPipeline::chunkCountFor(0, 256), std::invalid_argument);
    EXPECT_THROW(ChunkPipeline::chunkCountFor(-1, 256), std::invalid_argument);
    EXPECT_THROW(ChunkPipeline::chunkCountFor(256, 0), std::invalid_argument);
    EXPECT_THROW(ChunkPipeline::lastTokenIndexFor(0, 256), std::invalid_argument);
}
