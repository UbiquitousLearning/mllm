cmake_minimum_required(VERSION 3.11)
project(xxHash C)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i[3-6]86")
  option(XXH_ENABLE_DISPATCH "enable dispatch" ON)
else()
  set(XXH_ENABLE_DISPATCH OFF)
endif()

set(_src src/xxhash.c)
if(XXH_ENABLE_DISPATCH)
  list(APPEND _src src/xxh_x86dispatch.c)
endif()

add_library(xxhash INTERFACE)
add_library(xxHash::xxhash ALIAS xxhash)

target_include_directories(xxhash INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_library(xxhash_impl OBJECT ${_src})
target_include_directories(xxhash_impl PUBLIC include)
if(XXH_ENABLE_DISPATCH)
  target_compile_definitions(xxhash_impl PUBLIC XXH_DISPATCH_ENABLE=1)
endif()

target_link_libraries(xxhash INTERFACE xxhash_impl)

add_library(xxhash_static STATIC $<TARGET_OBJECTS:xxhash_impl>)
add_library(xxHash::xxhash ALIAS xxhash_static)
target_include_directories(xxhash_static PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
