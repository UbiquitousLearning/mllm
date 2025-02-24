macro(func_link_libaries target)
    target_link_libraries(${target} PUBLIC MLLM_CPU fmt::fmt-header-only)
    if (MLLM_OPENMP)
        target_compile_options(${target} PRIVATE -fopenmp)
        if (ARM)
            target_link_libraries(${target} PUBLIC -fopenmp -static-openmp)
        else ()
            target_link_libraries(${target} PUBLIC -fopenmp)
        endif ()
    endif ()
    if (QNN)
        target_compile_definitions(${target} PRIVATE USE_QNN)
        target_link_libraries(${target} PUBLIC MLLM_QNN ${CMAKE_DL_LIBS})
    endif()
    if (MLLM_BUILD_XNNPACK_BACKEND)
        target_link_libraries(${target} PRIVATE MllmXnnpackBackend)
    endif()
endmacro()

macro(func_llm_add_executable target)
    add_executable(${target}
        ${PROJECT_SOURCE_DIR}/examples/${target}.cpp 
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
    func_link_libaries(${target})
endmacro()

macro(func_vlm_add_executable target)
    add_executable(${target} 
        ${PROJECT_SOURCE_DIR}/examples/${target}.cpp 
        ${DIR_SRC_CPU} 
        ${DIR_SRC_MEM_MANAGER} 
        ${DIR_SRC_EXP} 
        ${DIR_SRC}
        ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/Unicode.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/UnicodeData.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/WordPiece/WordPiece.cpp
        ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
        ${DIR_SRC_PROCESSOE}
        ${DIR_THIRDPARTY_AUDIO}
    )
    func_link_libaries(${target})
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



func_vlm_add_executable(demo_llava)
func_vlm_add_executable(demo_fuyu)
func_vlm_add_executable(demo_vit)
func_vlm_add_executable(demo_clip)
func_vlm_add_executable(demo_imagebind)
func_vlm_add_executable(demo_imagebind_1mod)
func_vlm_add_executable(demo_phi3v)
func_vlm_add_executable(demo_qwen2_vl)
# func_vlm_add_executable(demo)

# QNN demo

if(QNN)
    func_llm_add_executable(demo_qwen_npu)
    func_llm_add_executable(main_qwen_npu)
    func_llm_add_executable(demo_phonelm_npu)
    func_llm_add_executable(main_phonelm_npu)
    func_llm_add_executable(demo_qwen2.5_npu)
    func_llm_add_executable(demo_qwen_pipeline)
endif()


if(MLLM_BUILD_XNNPACK_BACKEND)
    func_llm_add_executable(demo_qwen_xp)
endif()


# old main
# macro(func_o_vlm_add_executable target)
#     add_executable(${target} 
#         ${PROJECT_SOURCE_DIR}/examples/${target}.cpp 
#         ${DIR_SRC_CPU} 
#         ${DIR_SRC_MEM_MANAGER} 
#         ${DIR_SRC_EXP} 
#         ${DIR_SRC}
#         ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
#         ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
#         ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
#         ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
#         ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
#         ${PROJECT_SOURCE_DIR}/src/processor/FuyuPreProcess.cpp
#     )
#     func_link_libaries(${target})
# endmacro()
# macro(func_o_avlm_add_executable target)
#     add_executable(${target} 
#         ${PROJECT_SOURCE_DIR}/examples/${target}.cpp 
#         ${DIR_SRC_CPU} 
#         ${DIR_SRC_MEM_MANAGER} 
#         ${DIR_SRC_EXP} 
#         ${DIR_SRC}
#         ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
#         ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
#         ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
#         ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
#         ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
#         ${DIR_SRC_PROCESSOE}
#         ${DIR_THIRDPARTY_AUDIO}
#     )
#     func_link_libaries(${target})
# endmacro()
# func_llm_add_executable(main_llama)
# func_llm_add_executable(main_alpaca)
# func_llm_add_executable(main_tinyllama)
# func_o_vlm_add_executable(main_llava)
# func_o_vlm_add_executable(main_fuyu)
# func_o_vlm_add_executable(main_vit)
# func_o_vlm_add_executable(main_clip)
# func_o_avlm_add_executable(main_imagebind)
