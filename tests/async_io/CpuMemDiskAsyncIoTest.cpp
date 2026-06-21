#include <mllm/mllm.hpp>
#include <mllm/utils/Argparse.hpp>
#include <mllm/engine/io/Primitives.hpp>

MLLM_MAIN({
  auto args = mllm::engineArgAttach();
  mllm::Argparse::parse(argc, argv);
  mllm::configEngineWithArgs(args);

  // Enable CPU side Mem <==> Disk Async IO feature
  mllm::enableCpuMemDiskAsyncIOFeature();

  // Create Tensor
  auto A = mllm::Tensor::random({1024, 1024});
  auto B = mllm::Tensor::random({1024, 1024});
  auto B_copy = mllm::Tensor::emptyLike(B).alloc();

  // Now, we can copy data async
  auto future = mllm::async::io::copy(B_copy, B);
  auto C = A + B;
  mllm::async::wait(future);

  // Or load from Disk
  // auto future_1 = mllm::async::io::loadAnonymousMemoryTensorFromDisk(B_copy, "some_expert", "model.mllm");
  // mllm::async::wait(future);

  // Check if everything is correct
  MLLM_RT_ASSERT(mllm::test::allClose(B, B_copy));
});
