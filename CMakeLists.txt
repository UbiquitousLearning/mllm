cmake_minimum_required(VERSION 3.0)

project(mllm)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

# temp executable name
set(TEST_EXE main_test)

aux_source_directory(${PROJECT_SOURCE_DIR}/src  DIR_SRC)
# aux_source_directory(${PROJECT_SOURCE_DIR}/src/backends/cpu  DIR_SRC_CPU)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples  EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test  TEST_SRC)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdpartys/include)
# include_directories(${PROJECT_SOURCE_DIR}/src/backends/cpu)

add_executable(${TEST_EXE} ${PROJECT_SOURCE_DIR}/test/main_test.cpp ${DIR_SRC_CPU} ${DIR_SRC})

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/cpu)
target_link_libraries(${TEST_EXE} MLLM_CPU)