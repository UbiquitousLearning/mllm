add_executable(Mllm-Test-Plugin-HostTest HostTest.cpp)
target_link_libraries(Mllm-Test-Plugin-HostTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(Mllm-Test-Plugin-HostTest PRIVATE ${MLLM_INCLUDE_DIR})

add_library(CustomPackageForHostTest SHARED CustomPackageForHostTest.cpp)
target_link_libraries(CustomPackageForHostTest PRIVATE MllmRT MllmCPUBackend)
target_include_directories(CustomPackageForHostTest PRIVATE ${MLLM_INCLUDE_DIR})
