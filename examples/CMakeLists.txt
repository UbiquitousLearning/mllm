set(COMMON_SRC 
    ${DIR_SRC_CPU}
    ${DIR_SRC_MEM_MANAGER}
    ${DIR_SRC_EXP}
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/WordPiece/WordPiece.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tiktoken/tiktoken.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unicode.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/UnicodeData.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)

set(VLM_SPECIFIC_SRC
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
    ${DIR_SRC_PROCESSOE}
    ${DIR_THIRDPARTY_AUDIO}
)

macro(func_set_compile_opts_defs target)
    if (MLLM_OPENMP)
        target_compile_options(${target} PRIVATE -fopenmp)
    endif()
    if (QNN)
        target_compile_definitions(${target} PRIVATE USE_QNN)
    endif()
endmacro()

macro(func_link_libs target)
    target_link_libraries(${target} PUBLIC lib_mllm_cpu fmt::fmt-header-only)
    if (MLLM_OPENMP)
        if (ARM AND NOT (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_CROSSCOMPILING))
            # 非Mac的ARM，静态链接OpenMP
            target_link_libraries(${target} PUBLIC -fopenmp -static-openmp)
        else()
            # 其它平台（含Mac），动态链接OpenMP
            target_link_libraries(${target} PUBLIC -fopenmp)
        endif()
    endif()
    if (QNN)
        target_link_libraries(${target} PUBLIC lib_mllm_qnn ${CMAKE_DL_LIBS})
    endif()
    if (MLLM_BUILD_XNNPACK_BACKEND)
        target_link_libraries(${target} PRIVATE lib_mllm_xnnpack)
    endif()
endmacro()

add_library(lib_mllm_common_llm STATIC ${COMMON_SRC})
func_set_compile_opts_defs(lib_mllm_common_llm)

add_library(lib_mllm_vlm STATIC ${VLM_SPECIFIC_SRC})
target_link_libraries(lib_mllm_vlm PUBLIC lib_mllm_common_llm)
func_set_compile_opts_defs(lib_mllm_vlm)

macro(func_llm_add_executable target)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/examples/${target}.cpp)
    target_link_libraries(${target} PUBLIC lib_mllm_common_llm)
    func_set_compile_opts_defs(${target})
    func_link_libs(${target})
endmacro()

macro(func_vlm_add_executable target)
    add_executable(${target} ${PROJECT_SOURCE_DIR}/examples/${target}.cpp)
    target_link_libraries(${target} PUBLIC lib_mllm_vlm)
    func_set_compile_opts_defs(${target})
    func_link_libs(${target})
endmacro()

func_llm_add_executable(mllm_benchmark)
func_llm_add_executable(demo_llama)
func_llm_add_executable(demo_tinyllama)
func_llm_add_executable(demo_stablelm)
func_llm_add_executable(demo_sparse_llama)
func_llm_add_executable(demo_elastic_llama)
func_llm_add_executable(demo_gemma)
func_llm_add_executable(demo_gemma2)
func_llm_add_executable(demo_qwen)
func_llm_add_executable(demo_qwen3)
func_llm_add_executable(demo_ds_qwen2)
func_llm_add_executable(demo_mistral)
func_llm_add_executable(demo_yi)
func_llm_add_executable(demo_opt)
func_llm_add_executable(demo_phi3)
func_llm_add_executable(demo_minicpm)
func_llm_add_executable(demo_minicpm3)
func_llm_add_executable(demo_minicpm_moe)
func_llm_add_executable(demo_smollm)
func_llm_add_executable(demo_openelm)
func_llm_add_executable(demo_dclm)
func_llm_add_executable(demo_bert)
func_llm_add_executable(demo_phonelm)
func_llm_add_executable(demo_llama3)
func_llm_add_executable(demo_minicpm_moe_mbm)
func_llm_add_executable(demo_qwen_sd)
func_llm_add_executable(demo_minicpm_moe_mbp)

func_vlm_add_executable(demo_llava)
func_vlm_add_executable(demo_fuyu)
func_vlm_add_executable(demo_vit)
func_vlm_add_executable(demo_clip)
func_vlm_add_executable(demo_imagebind)
func_vlm_add_executable(demo_imagebind_1mod)
func_vlm_add_executable(demo_phi3v)
func_vlm_add_executable(demo_qwen2_vl)
func_vlm_add_executable(demo_showui)
func_vlm_add_executable(demo_qwen2_vl_vtp)
func_vlm_add_executable(demo_showui_vtp)

if(QNN)
    func_llm_add_executable(demo_qwen_npu)
    func_llm_add_executable(demo_phonelm_npu)
    func_llm_add_executable(demo_qwen2.5_npu)
    func_vlm_add_executable(demo_qwen2_vl_npu)
endif()

if(MLLM_BUILD_XNNPACK_BACKEND)
    func_llm_add_executable(demo_qwen_xp)
endif()