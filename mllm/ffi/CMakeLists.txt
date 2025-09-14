find_package(Python COMPONENTS Interpreter REQUIRED)
set(tvm_ffi_ROOT ${CMAKE_CURRENT_LIST_DIR}/vendors/tvm-ffi)
add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)

add_library(MllmFFIExtension SHARED ${CMAKE_CURRENT_LIST_DIR}/extension.cc)
target_link_libraries(MllmFFIExtension PUBLIC tvm_ffi_shared MllmRT)
