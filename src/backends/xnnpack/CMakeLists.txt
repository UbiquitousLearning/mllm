add_subdirectory(third_party/XNNPACK)

add_library(MllmXnnpackBackend
    OBJECT

    Utils/Logger.cpp

    XpInterface.cpp
    XnnpackBackend.cpp
    XpMemoryManager.cpp
    XpWrapper.cpp

    Ops/XpDirect.cpp
    Ops/XpBinary.cpp
    Ops/XpDispatch.cpp
    Ops/XpLinear.cpp
    Ops/XpMatmul.cpp
    Ops/XpRoPE.cpp
    Ops/XpSoftmax.cpp
    Ops/XpRMSNorm.cpp
    Ops/XpTranspose.cpp
    Ops/XpKVCache.cpp
    Ops/XpSiLU.cpp
    Ops/XpReLU.cpp
    Ops/XpGeLU.cpp
    Ops/XpSubGraphStart.cpp
    Ops/XpSubGraphFinalize.cpp
    Ops/XpD2H.cpp
    Ops/XpCausalMask.cpp
    Ops/XpSDPA.cpp
    Ops/XpEmbedding.cpp
    Ops/XpParameter.cpp

    Functions/XpBinaryFunc.cpp
    Functions/XpTransposeFunc.cpp
    Functions/XpViewFunc.cpp
    Functions/XpMatmulFunc.cpp
)
target_include_directories(MllmXnnpackBackend PUBLIC third_party/XNNPACK/src/)
target_include_directories(MllmXnnpackBackend PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../)
target_link_libraries(MllmXnnpackBackend PUBLIC XNNPACK fmt::fmt-header-only)
set_target_properties(MllmXnnpackBackend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
