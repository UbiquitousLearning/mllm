project(mllm)

set(CMAKE_CXX_STANDARD 14)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)

aux_source_directory(${PROJECT_SOURCE_DIR}/src  DIR_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/backends/cpu  DIR_SRC_CPU)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples  EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test  TEST_SRC)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdpartys/include)
# include_directories(${PROJECT_SOURCE_DIR}/src/backends/cpu)

add_executable(main_test ${TEST_SRC} ${DIR_SRC_CPU} ${DIR_SRC})