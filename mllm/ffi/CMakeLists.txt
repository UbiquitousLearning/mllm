find_package(Python COMPONENTS Interpreter REQUIRED)
set(tvm_ffi_ROOT ${CMAKE_CURRENT_LIST_DIR}/vendors/tvm-ffi)

if(ANDROID)
  # TVM's Libbacktrace has config error. We need to disable it.
  set(TVM_FFI_USE_LIBBACKTRACE OFF)
endif()
add_subdirectory(${tvm_ffi_ROOT} tvm_ffi)

add_library(MllmFFIExtension SHARED 
  ${CMAKE_CURRENT_LIST_DIR}/Extension.cc
  ${CMAKE_CURRENT_LIST_DIR}/Object.cc
)
target_link_libraries(MllmFFIExtension PUBLIC tvm_ffi_header)
target_link_libraries(MllmFFIExtension PUBLIC tvm_ffi_shared MllmRT)
set_target_properties(MllmFFIExtension PROPERTIES PREFIX "")
install(TARGETS MllmFFIExtension DESTINATION .)
