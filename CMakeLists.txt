cmake_minimum_required(VERSION 3.7.2)

project(mllm)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# temp executable name
set(TEST_EXE main_test)

# find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
#set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

# backend options
option(NNAPI "Enable NNAPI" OFF)

if(CMAKE_HOST_UNIX)
    message("current platform: Linux ")
#    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14 -stdlib=libc++")
elseif(CMAKE_HOST_WIN32)
    message("current platform: Windows ")
else()
    message("current platform: unkonw ")
endif()

aux_source_directory(${PROJECT_SOURCE_DIR}/src  DIR_SRC)
# aux_source_directory(${PROJECT_SOURCE_DIR}/src/backends/cpu  DIR_SRC_CPU)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/express  DIR_SRC_EXP)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples  EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test  TEST_SRC)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdpartys/include)
# include_directories(${PROJECT_SOURCE_DIR}/src/backends/cpu)

add_executable(${TEST_EXE} ${PROJECT_SOURCE_DIR}/test/main_test.cpp ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/cpu)
target_link_libraries(${TEST_EXE} MLLM_CPU)

if(NNAPI)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/nnapi)
    target_link_libraries(${TEST_EXE} MLLM_NNAPI)
endif()