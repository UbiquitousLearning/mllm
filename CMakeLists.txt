cmake_minimum_required(VERSION 3.10)

project(mllm)
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)
option(ARM "build on ARM" OFF)
if(MSVC)
    message(STATUS "Using MSVC as the compiler")
else()
    add_compile_options(-Wno-deprecated-declarations)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    set(ARM ON)
    set(ANDROID_PLATFORM android-28)

    # add_compile_options(-std=c++17)
endif ()

if (ARM)
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin-arm)
else ()
    set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../bin)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    option(TEST "test mode" OFF)
else ()
    option(TEST "test mode" ON)
endif ()
option(QUANT "quantize tools" ON)
option(APK "Build for Android APK Lib." OFF)
option(FROM_GGUF "convert from gguf" OFF)

if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif ()

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# find_program(CLANG_TIDY_EXE NAMES "clang-tidy" REQUIRED)
# set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
option(DEBUG "debug print" OFF)

if (CMAKE_HOST_UNIX)
    message(STATUS "current platform: Linux ")

    # set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14 -stdlib=libc++")
elseif (CMAKE_HOST_WIN32)
    message(STATUS "current platform: Windows ")
else ()
    message(STATUS "current platform: unknown ")
endif ()

aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRC)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/express DIR_SRC_EXP)

aux_source_directory(${PROJECT_SOURCE_DIR}/src/processor DIR_SRC_PROCESSOE)
aux_source_directory(${PROJECT_SOURCE_DIR}/src/memory DIR_SRC_MEM_MANAGER)
aux_source_directory(${PROJECT_SOURCE_DIR}/examples EMP_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/test TEST_SRC)
aux_source_directory(${PROJECT_SOURCE_DIR}/third_party/wenet_audio DIR_THIRDPARTY_AUDIO)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/src/backends/cpu)

#add tokenizers
file(GLOB_RECURSE SRC_TOKENIZERS
        ${PROJECT_SOURCE_DIR}/src/tokenizers/*.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/*.hpp
)
# if compile to x86_64
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86_64 detected")
    add_compile_options(-mf16c)
    add_compile_options(-mavx2)
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    add_definitions(-DARM)
    if (APK)
        message(STATUS "Build for Android APK Lib.")
        add_definitions("-DANDROID_API")
    endif ()
endif ()

if (QUANT)
    include_directories(${PROJECT_SOURCE_DIR}/src/quantizer)
    file(GLOB_RECURSE MLLM_QUANT

            ${PROJECT_SOURCE_DIR}/src/backends/cpu/quantize/*.hpp
            ${PROJECT_SOURCE_DIR}/src/backends/cpu/quantize/*.cpp
    )

    file(GLOB_RECURSE MLLM_QUANTIZER
            ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/*.hpp)
    list(REMOVE_ITEM MLLM_QUANTIZER ${CMAKE_CURRENT_LIST_DIR}/src/quantizer/main.cpp)

    message(STATUS "MLLM_Quant: ${MLLM_QUANT}")

    add_executable(
            quantize
            ${PROJECT_SOURCE_DIR}/src/quantizer/main.cpp
            ${MLLM_QUANT}
            ${MLLM_QUANTIZER}

            # ${DIR_SRC}
            ${PROJECT_SOURCE_DIR}/src/ParamLoader.cpp
    )
    if (FROM_GGUF)
        add_executable(
                from_gguf
                ${PROJECT_SOURCE_DIR}/tools/gguf_convertor/gguf.cpp
                ${PROJECT_SOURCE_DIR}/tools/gguf_convertor/gguf.hpp
                ${MLLM_QUANT}
                ${MLLM_QUANTIZER}
                # ${DIR_SRC}
                ${PROJECT_SOURCE_DIR}/src/ParamLoader.cpp

        )
    endif ()

endif ()

if (TEST)
    add_definitions(-DTEST)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
    include_directories(${PROJECT_SOURCE_DIR}/test)

    # MLLM Tests
    file(GLOB_RECURSE MLLM_TEST
            ${CMAKE_CURRENT_LIST_DIR}/test/*.cpp
            ${CMAKE_CURRENT_LIST_DIR}/test/*.hpp
    )
    list(REMOVE_ITEM MLLM_TEST ${CMAKE_CURRENT_LIST_DIR}/test/TestSystemMemoryManager.cpp)
    list(REMOVE_ITEM MLLM_TEST ${CMAKE_CURRENT_LIST_DIR}/test/TestMemoryPoolManager.cpp)
    # list(REMOVE_ITEM MLLM_TEST ${CMAKE_CURRENT_LIST_DIR}/test/clip_tokenizer_test.cpp)

    message(STATUS "MLLM_TEST: ${MLLM_TEST}")

    # Add ${DIR_SRC_QUANT} to ${DIR_SRC}
    # if ()
    add_executable(
            MLLM_TEST
            ${PROJECT_SOURCE_DIR}/test/main.cpp
            ${MLLM_TEST}
            ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} ${MLLM_QUANTIZER} ${SRC_TOKENIZERS}
            src/processor/ClipPreProcess.hpp
            src/processor/ClipPreProcess.cpp
            src/processor/FuyuPreProcess.hpp
            src/processor/FuyuPreProcess.cpp
            src/processor/PreProcess.hpp
            src/processor/PreProcess.cpp
            test/processor/ClipPreprocessorTest.cpp
    )

    target_link_libraries(
            MLLM_TEST
            GTest::gtest_main
            MLLM_CPU
    )
    add_executable(
            memoryPoolTest
            ${PROJECT_SOURCE_DIR}/test/TestMemoryPoolManager.cpp
            ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
    )
    add_executable(
            SystemMemoryTest
            ${PROJECT_SOURCE_DIR}/test/TestSystemMemoryManager.cpp
            ${DIR_SRC_MEM_MANAGER} ${PROJECT_SOURCE_DIR}/src/MemoryManager.hpp
    )
    include(GoogleTest)

    # if not cross-compiling
    if (NOT ARM)
        gtest_discover_tests(MLLM_TEST)
    endif ()
endif ()

# else ()
# No Unit Test.
add_executable(main_llama ${PROJECT_SOURCE_DIR}/examples/main_llama.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
)
if (ARM AND NOT APK)
    target_compile_options(main_llama PRIVATE -fopenmp)
    target_link_libraries(main_llama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_llama MLLM_CPU)
endif ()

add_executable(main_alpaca ${PROJECT_SOURCE_DIR}/examples/main_alpaca.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
)
if (ARM AND NOT APK)
    target_compile_options(main_alpaca PRIVATE -fopenmp)
    target_link_libraries(main_alpaca PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_alpaca MLLM_CPU)
endif ()

add_executable(main_llava ${PROJECT_SOURCE_DIR}/examples/main_llava.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
        src/processor/ClipPreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(main_llava PRIVATE -fopenmp)
    target_link_libraries(main_llava PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_llava MLLM_CPU)
endif ()


add_executable(main_tinyllama ${PROJECT_SOURCE_DIR}/examples/main_tinyllama.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
)
if (ARM AND NOT APK)
    target_compile_options(main_tinyllama PRIVATE -fopenmp)
    target_link_libraries(main_tinyllama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_tinyllama MLLM_CPU)
endif ()

add_executable(main_fuyu ${PROJECT_SOURCE_DIR}/examples/main_fuyu.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/Unigram/Unigram.hpp
        src/tokenizers/Unigram/Unigram.cpp
        src/tokenizers/Unigram/trie.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
        src/processor/PreProcess.hpp
        src/processor/FuyuPreProcess.hpp
        src/processor/FuyuPreProcess.cpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(main_fuyu PRIVATE -fopenmp)
    target_link_libraries(main_fuyu PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_fuyu MLLM_CPU)
endif ()

add_executable(main_vit ${PROJECT_SOURCE_DIR}/examples/main_vit.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/Unigram/Unigram.hpp
        src/tokenizers/Unigram/Unigram.cpp
        src/tokenizers/Unigram/trie.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
        src/processor/PreProcess.hpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(main_vit PRIVATE -fopenmp)
    target_link_libraries(main_vit PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_vit MLLM_CPU)
endif ()

add_executable(main_clip ${PROJECT_SOURCE_DIR}/examples/main_clip.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
        src/processor/PreProcess.hpp
        src/processor/PreProcess.cpp
        src/processor/ClipPreProcess.cpp
        src/processor/ClipPreProcess.hpp
)
if (ARM AND NOT APK)
    target_compile_options(main_clip PRIVATE -fopenmp)
    target_link_libraries(main_clip PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_clip MLLM_CPU)
endif ()

add_executable(main_imagebind ${PROJECT_SOURCE_DIR}/examples/main_imagebind.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
        ${DIR_SRC_PROCESSOE}
        ${DIR_THIRDPARTY_AUDIO}
        #        src/processor/ClipPreProcess.cpp
        #        src/processor/ClipPreProcess.hpp
        src/processor/PreProcess.hpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(main_imagebind PRIVATE -fopenmp)
    target_link_libraries(main_imagebind PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(main_imagebind MLLM_CPU)
endif ()



add_executable(demo_llama ${PROJECT_SOURCE_DIR}/examples/demo_llama.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_llama PRIVATE -fopenmp)
    target_link_libraries(demo_llama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_llama MLLM_CPU)
endif ()



add_executable(demo_fuyu ${PROJECT_SOURCE_DIR}/examples/demo_fuyu.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Unigram/Unigram.cpp
        src/processor/FuyuPreProcess.cpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_fuyu PRIVATE -fopenmp)
    target_link_libraries(demo_fuyu PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_fuyu MLLM_CPU)
endif ()


add_executable(demo_vit ${PROJECT_SOURCE_DIR}/examples/demo_vit.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_vit PRIVATE -fopenmp)
    target_link_libraries(demo_vit PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_vit MLLM_CPU)
endif ()

add_executable(demo_clip ${PROJECT_SOURCE_DIR}/examples/demo_clip.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
        src/processor/ClipPreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_clip PRIVATE -fopenmp)
    target_link_libraries(demo_clip PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_clip MLLM_CPU)
endif ()

add_executable(demo_imagebind ${PROJECT_SOURCE_DIR}/examples/demo_imagebind.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        ${DIR_SRC_PROCESSOE}
        ${DIR_THIRDPARTY_AUDIO}
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_imagebind PRIVATE -fopenmp)
    target_link_libraries(demo_imagebind PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_imagebind MLLM_CPU)
endif ()

add_executable(demo_tinyllama ${PROJECT_SOURCE_DIR}/examples/demo_tinyllama.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/Tokenizer.hpp
        src/tokenizers/BPE/Bpe.cpp
        src/tokenizers/BPE/Bpe.hpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_tinyllama PRIVATE -fopenmp)
    target_link_libraries(demo_tinyllama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_tinyllama MLLM_CPU)
endif ()

add_executable(demo_llava ${PROJECT_SOURCE_DIR}/examples/demo_llava.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
        src/processor/ClipPreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_llava PRIVATE -fopenmp)
    target_link_libraries(demo_llava PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_llava MLLM_CPU)
endif ()

add_executable(demo_gemma ${PROJECT_SOURCE_DIR}/examples/demo_gemma.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_gemma PRIVATE -fopenmp)
    target_link_libraries(demo_gemma PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_gemma MLLM_CPU)
endif ()

add_executable(demo_qwen ${PROJECT_SOURCE_DIR}/examples/demo_qwen.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_qwen PRIVATE -fopenmp)
    target_link_libraries(demo_qwen PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_qwen PUBLIC MLLM_CPU)
endif ()

add_executable(demo_mistral ${PROJECT_SOURCE_DIR}/examples/demo_mistral.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_mistral PRIVATE -fopenmp)
    target_link_libraries(demo_mistral PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_mistral PUBLIC MLLM_CPU)
endif ()

add_executable(demo_yi ${PROJECT_SOURCE_DIR}/examples/demo_yi.cpp ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} ${DIR_SRC_EXP} ${DIR_SRC}
        src/tokenizers/Tokenizer.cpp
        src/tokenizers/BPE/Bpe.cpp
        src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_yi PRIVATE -fopenmp)
    target_link_libraries(demo_yi PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_link_libraries(demo_yi PUBLIC MLLM_CPU)
endif ()

if (APK)
    add_library(mllm_lib STATIC ${DIR_SRC_CPU} ${DIR_SRC_EXP} ${DIR_SRC} ${DIR_SRC_MEM_MANAGER}
            src/tokenizers/Tokenizer.cpp
            src/tokenizers/Tokenizer.hpp
            src/tokenizers/Unigram/Unigram.hpp
            src/tokenizers/Unigram/Unigram.cpp
            src/tokenizers/Unigram/trie.hpp
            src/tokenizers/BPE/Bpe.cpp
            src/tokenizers/BPE/Bpe.hpp
            src/processor/PreProcess.hpp
            src/processor/FuyuPreProcess.hpp
            src/processor/FuyuPreProcess.cpp
            tools/jni/LibHelper.cpp
            tools/jni/LibHelper.hpp
            tools/jni/modeling_llama.hpp
            tools/jni/modeling_fuyu.hpp
            tools/jni/helper.hpp
            src/processor/PreProcess.cpp

    )
    target_link_libraries(mllm_lib MLLM_CPU)

endif ()
