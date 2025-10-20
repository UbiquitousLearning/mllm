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
  ${CMAKE_CURRENT_LIST_DIR}/ModelService.cc
  ${CMAKE_CURRENT_LIST_DIR}/Nn.cc
  ${CMAKE_CURRENT_LIST_DIR}/Compile.cc
)
target_link_libraries(MllmFFIExtension PUBLIC tvm_ffi_header)
target_link_libraries(MllmFFIExtension PUBLIC tvm_ffi_shared MllmRT MllmCPUBackend)
set_target_properties(MllmFFIExtension PROPERTIES PREFIX "")

# Set the depend search path. Windows do not need this, it will search dlls in the same directory first.
if(APPLE)
    set_target_properties(MllmFFIExtension PROPERTIES
        INSTALL_RPATH "@loader_path"
        MACOSX_RPATH ON)
elseif(UNIX)   # Linux
    set_target_properties(MllmFFIExtension PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()
install(
  TARGETS MllmFFIExtension tvm_ffi_header tvm_ffi_shared
  EXPORT MllmTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)
