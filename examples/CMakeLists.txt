# cpu executables
if(NOT QNN) 
add_executable(main_llama ${PROJECT_SOURCE_DIR}/examples/main_llama.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_llama PRIVATE -fopenmp)
    target_link_libraries(main_llama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_llama PRIVATE -fopenmp)
    target_link_libraries(main_llama PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(main_alpaca 
    ${PROJECT_SOURCE_DIR}/examples/main_alpaca.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_alpaca PRIVATE -fopenmp)
    target_link_libraries(main_alpaca PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_alpaca PRIVATE -fopenmp)
    target_link_libraries(main_alpaca PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(main_llava 
    ${PROJECT_SOURCE_DIR}/examples/main_llava.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_llava PRIVATE -fopenmp)
    target_link_libraries(main_llava PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_llava PRIVATE -fopenmp)
    target_link_libraries(main_llava PUBLIC MLLM_CPU -fopenmp)
endif ()


add_executable(main_tinyllama 
    ${PROJECT_SOURCE_DIR}/examples/main_tinyllama.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
)

if (MLLM_OPENMP_STATIC)
    target_compile_options(main_tinyllama PRIVATE -fopenmp)
    target_link_libraries(main_tinyllama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_tinyllama PRIVATE -fopenmp)
    target_link_libraries(main_tinyllama PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(main_fuyu 
    ${PROJECT_SOURCE_DIR}/examples/main_fuyu.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/trie.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/FuyuPreProcess.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/FuyuPreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_fuyu PRIVATE -fopenmp)
    target_link_libraries(main_fuyu PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_fuyu PRIVATE -fopenmp)
    target_link_libraries(main_fuyu PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(main_vit 
    ${PROJECT_SOURCE_DIR}/examples/main_vit.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/trie.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_vit PRIVATE -fopenmp)
    target_link_libraries(main_vit PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_vit PRIVATE -fopenmp)
    target_link_libraries(main_vit PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(main_clip 
    ${PROJECT_SOURCE_DIR}/examples/main_clip.cpp 
    ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.hpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_clip PRIVATE -fopenmp)
    target_link_libraries(main_clip PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_clip PRIVATE -fopenmp)
    target_link_libraries(main_clip PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(main_imagebind 
    ${PROJECT_SOURCE_DIR}/examples/main_imagebind.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
    ${DIR_SRC_PROCESSOE}
    ${DIR_THIRDPARTY_AUDIO}
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.hpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(main_imagebind PRIVATE -fopenmp)
    target_link_libraries(main_imagebind PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(main_imagebind PRIVATE -fopenmp)
    target_link_libraries(main_imagebind PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_stablelm 
    ${PROJECT_SOURCE_DIR}/examples/demo_stablelm.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_stablelm PRIVATE -fopenmp)
    target_link_libraries(demo_stablelm PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_stablelm PRIVATE -fopenmp)
    target_link_libraries(demo_stablelm PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_llama 
    ${PROJECT_SOURCE_DIR}/examples/demo_llama.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_llama PRIVATE -fopenmp)
    target_link_libraries(demo_llama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_llama PRIVATE -fopenmp)
    target_link_libraries(demo_llama PUBLIC MLLM_CPU -fopenmp)
endif ()



add_executable(demo_fuyu 
    ${PROJECT_SOURCE_DIR}/examples/demo_fuyu.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Unigram/Unigram.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/FuyuPreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_fuyu PRIVATE -fopenmp)
    target_link_libraries(demo_fuyu PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_fuyu PRIVATE -fopenmp)
    target_link_libraries(demo_fuyu PUBLIC MLLM_CPU -fopenmp)
endif ()


add_executable(demo_vit 
    ${PROJECT_SOURCE_DIR}/examples/demo_vit.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_vit PRIVATE -fopenmp)
    target_link_libraries(demo_vit PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_vit PRIVATE -fopenmp)
    target_link_libraries(demo_vit PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_clip 
    ${PROJECT_SOURCE_DIR}/examples/demo_clip.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_clip PRIVATE -fopenmp)
    target_link_libraries(demo_clip PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_clip PRIVATE -fopenmp)
    target_link_libraries(demo_clip PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_imagebind 
    ${PROJECT_SOURCE_DIR}/examples/demo_imagebind.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${DIR_SRC_PROCESSOE}
    ${DIR_THIRDPARTY_AUDIO}
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_imagebind PRIVATE -fopenmp)
    target_link_libraries(demo_imagebind PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_imagebind PRIVATE -fopenmp)
    target_link_libraries(demo_imagebind PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_imagebind_1mod 
    ${PROJECT_SOURCE_DIR}/examples/demo_imagebind_1mod.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${DIR_SRC_PROCESSOE}
    ${DIR_THIRDPARTY_AUDIO}
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_imagebind_1mod PRIVATE -fopenmp)
    target_link_libraries(demo_imagebind_1mod PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_imagebind_1mod PRIVATE -fopenmp)
    target_link_libraries(demo_imagebind_1mod PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_tinyllama 
    ${PROJECT_SOURCE_DIR}/examples/demo_tinyllama.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_tinyllama PRIVATE -fopenmp)
    target_link_libraries(demo_tinyllama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_tinyllama PRIVATE -fopenmp)
    target_link_libraries(demo_tinyllama PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_sparse_llama 
    ${PROJECT_SOURCE_DIR}/examples/demo_sparse_llama.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_sparse_llama PRIVATE -fopenmp)
    target_link_libraries(demo_sparse_llama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_sparse_llama PRIVATE -fopenmp)
    target_link_libraries(demo_sparse_llama PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_elastic_llama 
    ${PROJECT_SOURCE_DIR}/examples/demo_elastic_llama.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC} # ${DIR_SRC_QUANT}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.hpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.hpp
)
# target_compile_definitions(demo_elastic_llama PRIVATE MLLM_QKK_64)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_elastic_llama PRIVATE -fopenmp)
    target_link_libraries(demo_elastic_llama PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_elastic_llama PRIVATE -fopenmp)
    target_link_libraries(demo_elastic_llama PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_llava 
    ${PROJECT_SOURCE_DIR}/examples/demo_llava.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/ClipPreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_llava PRIVATE -fopenmp)
    target_link_libraries(demo_llava PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_llava PRIVATE -fopenmp)
    target_link_libraries(demo_llava PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_gemma 
    ${PROJECT_SOURCE_DIR}/examples/demo_gemma.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_gemma PRIVATE -fopenmp)
    target_link_libraries(demo_gemma PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_gemma PRIVATE -fopenmp)
    target_link_libraries(demo_gemma PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_qwen 
    ${PROJECT_SOURCE_DIR}/examples/demo_qwen.cpp 
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
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_qwen PRIVATE -fopenmp)
    target_link_libraries(demo_qwen PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    # target_link_libraries(demo_qwen PUBLIC MLLM_CPU)
    target_compile_options(demo_qwen PRIVATE -fopenmp)
    target_link_libraries(demo_qwen PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_mistral 
    ${PROJECT_SOURCE_DIR}/examples/demo_mistral.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_mistral PRIVATE -fopenmp)
    target_link_libraries(demo_mistral PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_mistral PRIVATE -fopenmp)
    target_link_libraries(demo_mistral PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_yi 
    ${PROJECT_SOURCE_DIR}/examples/demo_yi.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_yi PRIVATE -fopenmp)
    target_link_libraries(demo_yi PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_yi PRIVATE -fopenmp)
    target_link_libraries(demo_yi PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_opt 
    ${PROJECT_SOURCE_DIR}/examples/demo_opt.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_opt PRIVATE -fopenmp)
    target_link_libraries(demo_opt PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_opt PRIVATE -fopenmp)
    target_link_libraries(demo_opt PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_phi3 
    ${PROJECT_SOURCE_DIR}/examples/demo_phi3.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
)
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_phi3 PRIVATE -fopenmp)
    target_link_libraries(demo_phi3 PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_phi3 PRIVATE -fopenmp)
    target_link_libraries(demo_phi3 PUBLIC MLLM_CPU -fopenmp)
endif ()


add_executable(demo_minicpm 
    ${PROJECT_SOURCE_DIR}/examples/demo_minicpm.cpp 
    ${DIR_SRC_CPU} 
    ${DIR_SRC_MEM_MANAGER} 
    ${DIR_SRC_EXP} 
    ${DIR_SRC}
    ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
    ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
)
if (ARM AND NOT APK)
    target_compile_options(demo_minicpm PRIVATE -fopenmp)
    target_link_libraries(demo_minicpm PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_minicpm PRIVATE -fopenmp)
    target_link_libraries(demo_minicpm PUBLIC MLLM_CPU -fopenmp)
endif ()

endif() # end of cpu executables

add_executable(demo_smollm
    ${PROJECT_SOURCE_DIR}/examples/demo_smollm.cpp 
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
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_smollm PRIVATE -fopenmp)
    target_link_libraries(demo_smollm PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_smollm PRIVATE -fopenmp)
    target_link_libraries(demo_smollm PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_openelm
    ${PROJECT_SOURCE_DIR}/examples/demo_openelm.cpp 
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
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_openelm PRIVATE -fopenmp)
    target_link_libraries(demo_openelm PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_openelm PRIVATE -fopenmp)
    target_link_libraries(demo_openelm PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(demo_dclm
    ${PROJECT_SOURCE_DIR}/examples/demo_dclm.cpp 
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
if (MLLM_OPENMP_STATIC)
    target_compile_options(demo_dclm PRIVATE -fopenmp)
    target_link_libraries(demo_dclm PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(demo_dclm PRIVATE -fopenmp)
    target_link_libraries(demo_dclm PUBLIC MLLM_CPU -fopenmp)
endif ()

add_executable(benchmark_llm
    ${PROJECT_SOURCE_DIR}/examples/benchmark.cpp 
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
if (MLLM_OPENMP_STATIC)
    target_compile_options(benchmark_llm PRIVATE -fopenmp)
    target_link_libraries(benchmark_llm PUBLIC MLLM_CPU -fopenmp -static-openmp)
else ()
    target_compile_options(benchmark_llm PRIVATE -fopenmp)
    target_link_libraries(benchmark_llm PUBLIC MLLM_CPU -fopenmp)
endif ()

# QNN demo
if(QNN)
    # qnn executables
    add_executable(main_qwen_npu 
        ${PROJECT_SOURCE_DIR}/examples/main_qwen_npu.cpp 
        ${DIR_SRC_CPU} ${DIR_SRC_MEM_MANAGER} 
        ${DIR_SRC_EXP} ${DIR_SRC}
        ${PROJECT_SOURCE_DIR}/src/tokenizers/Tokenizer.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/Unicode.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/UnicodeData.cpp
        ${PROJECT_SOURCE_DIR}/src/tokenizers/BPE/Bpe.cpp
        ${PROJECT_SOURCE_DIR}/src/processor/PreProcess.cpp
    )
    target_compile_definitions(main_qwen_npu PRIVATE USE_QNN)
    if (ARM)
        target_compile_options(main_qwen_npu PRIVATE -fopenmp)
        target_link_libraries(main_qwen_npu PUBLIC  MLLM_CPU MLLM_QNN ${CMAKE_DL_LIBS} -fopenmp -static-openmp)
    endif ()
endif()