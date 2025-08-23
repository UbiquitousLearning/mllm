add_executable(Mllm-Benchmark-EXT-STL-IntrusivePtr intrusive_ptr.cpp)
target_link_libraries(Mllm-Benchmark-EXT-STL-IntrusivePtr PRIVATE benchmark::benchmark MllmRT MllmCPUBackend)
