add_executable(ascend_add_demo main.cpp)

if(DEFINED ENV{ASCEND_HOME_PATH})
  target_include_directories(ascend_add_demo PRIVATE $ENV{ASCEND_HOME_PATH}/include)
  target_link_directories(ascend_add_demo PRIVATE $ENV{ASCEND_HOME_PATH}/lib64)
endif()

target_link_libraries(ascend_add_demo PRIVATE
  MllmRT
  MllmAscendBackend
  ascendcl  # 添加 ACL 库，因为 main.cpp 中直接使用了 aclrtMemcpy
)

set_target_properties(ascend_add_demo PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)

target_include_directories(ascend_add_demo PRIVATE
  ${CMAKE_SOURCE_DIR}
)

