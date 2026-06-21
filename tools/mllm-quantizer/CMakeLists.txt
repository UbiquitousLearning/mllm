if(MLLM_BUILD_ARM_BACKEND)
  add_executable(mllm-quantizer 
    main.cpp
    driver.cpp
    schema/gguf.cpp
    schema/kai.cpp
  )
else()
  add_executable(mllm-quantizer 
    main.cpp
    driver.cpp
    schema/gguf.cpp
  )
endif()
target_link_libraries(mllm-quantizer PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-quantizer PRIVATE ${MLLM_INCLUDE_DIR})
