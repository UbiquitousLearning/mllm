if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

message(STATUS "SOC_VERSION is ${SOC_VERSION}, RUN_MODE is ${RUN_MODE}")

# ============ The CATLASS Code ============
# TODO add catlass

# ============ MLLM Code ============
file(GLOB MLLM_ASCEND_CORE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Ascend*.cpp)
file(GLOB MLLM_ASCEND_MEMORY_FILES ${CMAKE_CURRENT_SOURCE_DIR}/memory/*.cpp)
file(GLOB MLLM_ASCEND_OPS_FILES ${CMAKE_CURRENT_SOURCE_DIR}/ops/*.cpp)

set(MLLM_ASCEND_SOURCES
  ${MLLM_ASCEND_CORE_FILES}
  ${MLLM_ASCEND_MEMORY_FILES}
  ${MLLM_ASCEND_OPS_FILES}
  ${CMAKE_CURRENT_SOURCE_DIR}/Register.cpp
)

add_library(MllmAscendBackend SHARED ${MLLM_ASCEND_SOURCES})

if(DEFINED ENV{ASCEND_HOME_PATH})
  target_include_directories(MllmAscendBackend PUBLIC $ENV{ASCEND_HOME_PATH}/include)
  target_link_directories(MllmAscendBackend PUBLIC $ENV{ASCEND_HOME_PATH}/lib64)
else()
  message(WARNING "ASCEND_HOME_PATH is not set, Ascend headers and libs may not be found")
endif()

if(DEFINED ENV{ATB_HOME_PATH})
  target_include_directories(MllmAscendBackend PUBLIC $ENV{ATB_HOME_PATH}/include)
  target_link_directories(MllmAscendBackend PUBLIC $ENV{ATB_HOME_PATH}/lib)
elseif(EXISTS "${PROJECT_SOURCE_DIR}/../libs/atb")
  message(STATUS "Found ATB in ${PROJECT_SOURCE_DIR}/../libs/atb")
  target_include_directories(MllmAscendBackend PUBLIC "${PROJECT_SOURCE_DIR}/../libs/atb/include")
  target_link_directories(MllmAscendBackend PUBLIC "${PROJECT_SOURCE_DIR}/../libs/atb/lib")
else()
  message(WARNING "ATB_HOME_PATH not defined, ATB library will not be linked")
endif()


target_link_libraries(MllmAscendBackend PUBLIC
  ascendcl
  opapi
  nnopbase
  atb
  MllmRT
)
target_compile_definitions(MllmAscendBackend PUBLIC ASCENDC_DUMP=0)

target_compile_definitions(MllmAscendBackend PUBLIC MLLM_USE_ASCEND_MEMPOOL)
set_target_properties(MllmAscendBackend PROPERTIES CXX_STANDARD 20)
target_compile_options(MllmAscendBackend PUBLIC 
  $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>> 
  -O2
)
target_include_directories(MllmAscendBackend PUBLIC "${MLLM_ASCEND_INSTALL_PATH}/include/")


