# 查找 OpenCL 库
find_package(OpenCL REQUIRED)
if(OpenCL_FOUND)
    message(STATUS "Found OpenCL library: ${OpenCL_LIBRARIES}")
    message(STATUS "Found OpenCL include directory: ${OpenCL_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCL library not found.")
endif()

# 包含 OpenCL 头文件目录
include_directories(${OpenCL_INCLUDE_DIRS})

# 收集所有 OpenCL 源文件
file(GLOB OPENCL_SRC
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/op/*.cpp
)

# 定义 OpenCL 后端库
add_library(mllm_opencl STATIC
    ${OPENCL_SRC}
)

# 链接 OpenCL 库
target_link_libraries(mllm_opencl PRIVATE OpenCL::OpenCL)

# 将 mllm_opencl 库的头文件目录设为 public
# 这样其他目标链接到 mllm_opencl 时就能自动找到头文件
target_include_directories(mllm_opencl PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)