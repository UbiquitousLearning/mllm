macro(func_link_libaries target)
    if (MLLM_OPENMP_STATIC)
        target_compile_options(${target} PRIVATE -fopenmp)
        target_link_libraries(${target} PUBLIC MLLM_CPU -fopenmp -static-openmp)
    else ()
        target_compile_options(${target} PRIVATE -fopenmp)
        target_link_libraries(${target} PUBLIC MLLM_CPU -fopenmp)
    endif ()
    if(QNN)
        target_compile_definitions(${target} PRIVATE USE_QNN)
        if (ARM)
            target_link_libraries(${target} PUBLIC  MLLM_CPU MLLM_QNN ${CMAKE_DL_LIBS} -fopenmp -static-openmp)
        endif ()
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
        ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
        ${DIR_SRC_PROCESSOE}
        ${DIR_THIRDPARTY_AUDIO}
    )
    func_link_libaries(${target})
endmacro()



## new demos

func_llm_add_executable(benchmark)
func_llm_add_executable(demo_llama)
func_llm_add_executable(demo_tinyllama)
func_llm_add_executable(demo_stablelm)
func_llm_add_executable(demo_sparse_llama)
func_llm_add_executable(demo_elastic_llama)
func_llm_add_executable(demo_gemma)
func_llm_add_executable(demo_qwen)
func_llm_add_executable(demo_mistral)
func_llm_add_executable(demo_yi)
func_llm_add_executable(demo_opt)
func_llm_add_executable(demo_phi3)
func_llm_add_executable(demo_minicpm)
func_llm_add_executable(demo_smollm)
func_llm_add_executable(demo_openelm)
func_llm_add_executable(demo_dclm)
func_llm_add_executable(demo_phonelm)

func_vlm_add_executable(demo_llava)
func_vlm_add_executable(demo_fuyu)
func_vlm_add_executable(demo_vit)
func_vlm_add_executable(demo_clip)
func_vlm_add_executable(demo_imagebind)
func_vlm_add_executable(demo_imagebind_1mod)
# func_vlm_add_executable(demo)

# QNN demo
if(QNN)
    func_llm_add_executable(main_qwen_npu)
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
