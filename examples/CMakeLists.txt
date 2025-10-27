set(COMMON_SRC 
    ${DIR_SRC_CPU}
    ${DIR_SRC_MEM_MANAGER}
    ${DIR_SRC_EXP}
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/WordPiece/WordPiece.cpp
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/Tiktoken/tiktoken.cpp
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/Unicode.cpp
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/UnicodeData.cpp
    ${PROJECT_SOURCE_DIR}/mllm/processor/PreProcess.cpp
)

set(VLM_SPECIFIC_SRC
    ${PROJECT_SOURCE_DIR}/mllm/tokenizers/Unigram/Unigram.cpp
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
    target_link_libraries(${target} PUBLIC mllm_cpu fmt)
    if (MLLM_OPENMP)
        if (ARM AND NOT (CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND NOT CMAKE_CROSSCOMPILING))
            # 非Mac的ARM，静态链接OpenMP
            target_link_libraries(${target} PUBLIC -fopenmp -static-openmp)
        else()
            # 其它平台（含Mac），动态链接OpenMP
            target_link_libraries(${target} PUBLIC -fopenmp)
        endif()
    endif()
    if(OPENCL)
        target_link_libraries(${target} PUBLIC mllm_opencl ${CMAKE_DL_LIBS})
    endif()
    if (QNN)
        target_link_libraries(${target} PUBLIC mllm_qnn ${CMAKE_DL_LIBS})
    endif()
    if (MLLM_BUILD_XNNPACK_BACKEND)
        target_link_libraries(${target} PRIVATE mllm_xnnpack)
    endif()
endmacro()

add_library(mllm_llm STATIC ${COMMON_SRC})
func_set_compile_opts_defs(mllm_llm)

add_library(mllm_vlm STATIC ${VLM_SPECIFIC_SRC})
target_link_libraries(mllm_vlm PUBLIC mllm_llm)
func_set_compile_opts_defs(mllm_vlm)

macro(func_llm_add_executable target)
    if(EXISTS "${PROJECT_SOURCE_DIR}/examples/${target}.cpp")
        add_executable(${target} ${PROJECT_SOURCE_DIR}/examples/${target}.cpp)
        target_link_libraries(${target} PUBLIC mllm_llm)
        func_set_compile_opts_defs(${target})
        func_link_libs(${target})
    else()
        message(WARNING "Skip ${target}: ${PROJECT_SOURCE_DIR}/examples/${target}.cpp not found")
    endif()
endmacro()

macro(func_vlm_add_executable target)
    if(EXISTS "${PROJECT_SOURCE_DIR}/examples/${target}.cpp")
        add_executable(${target} ${PROJECT_SOURCE_DIR}/examples/${target}.cpp)
        target_link_libraries(${target} PUBLIC mllm_vlm)
        func_set_compile_opts_defs(${target})
        func_link_libs(${target})
    else()
        message(WARNING "Skip ${target}: ${PROJECT_SOURCE_DIR}/examples/${target}.cpp not found")
    endif()
endmacro()

func_llm_add_executable(test)
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
func_llm_add_executable(demo_qwen_batch)
func_llm_add_executable(demo_minicpm_moe_mbp)
func_llm_add_executable(demo_bailing_moe)
func_llm_add_executable(demo_bailing_moe2)
func_llm_add_executable(demo_bailing_moe_mbp)
func_llm_add_executable(demo_bailing_moe2_mbp)
func_llm_add_executable(demo_smallthinker)
func_llm_add_executable(demo_smallthinker_mbp)

func_vlm_add_executable(demo_llava)
func_vlm_add_executable(demo_fuyu)
func_vlm_add_executable(demo_vit)
func_vlm_add_executable(demo_clip)
func_vlm_add_executable(demo_imagebind)
func_vlm_add_executable(demo_imagebind_1mod)
func_vlm_add_executable(demo_phi3v)
func_vlm_add_executable(demo_qwen2_vl)
func_vlm_add_executable(demo_qwen2.5_vl)
func_vlm_add_executable(demo_showui)
func_vlm_add_executable(demo_qwen2_vl_vtp)
func_vlm_add_executable(demo_showui_vtp)

if(QNN)
    # func_llm_add_executable(demo_phonelm_npu)
    func_llm_add_executable(demo_qwen_npu)
    # func_llm_add_executable(demo_qwen_npu_pipeline)
    func_vlm_add_executable(demo_qwen2_vl_npu)
endif()


if(MLLM_BUILD_XNNPACK_BACKEND)
    func_llm_add_executable(demo_qwen_xp)
endif()