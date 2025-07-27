# ======================================================================
#         最终修正版 - 匹配您当前的目录结构
# ======================================================================

# 收集所有 OpenCL 源文件
file(GLOB OPENCL_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
)

# 定义 OpenCL 后端静态库
add_library(mllm_opencl STATIC ${OPENCL_SRC})


if(ANDROID)
     # --- 安卓平台的逻辑 ---
    # 在安卓上，我们在 C++ 代码中通过 dlopen 动态加载 libOpenCL.so。
    # 因此，在编译时不应该链接 -lOpenCL。
    # 我们唯一需要的链接依赖是 dl 库，用于 dlopen/dlsym。
    # 这个依赖已经在 examples/CMakeLists.txt 中的 ${CMAKE_DL_LIBS} 变量里处理了。
    # 所以，这里什么都不用做。
    message(STATUS "OpenCL backend for Android: dynamic loading enabled, skipping link-time dependency on OpenCL.")

else()
    # --- 非安卓平台的逻辑 (macOS, Linux) ---
    find_package(OpenCL REQUIRED)
    target_include_directories(mllm_opencl PUBLIC ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(mllm_opencl PUBLIC OpenCL::OpenCL)
endif()

# 将 mllm_opencl 库自身的头文件目录设为 public
target_include_directories(mllm_opencl PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)