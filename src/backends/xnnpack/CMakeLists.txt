add_subdirectory(third_party/XNNPACK)
add_subdirectory(third_party/fmt EXCLUDE_FROM_ALL)

add_library(MllmXnnpackBackend
    OBJECT

    Utils/Logger.cpp

    XpInterface.cpp

    Ops/XpDirect.cpp
    Ops/XpBinary.cpp
    Ops/XpDispatch.cpp
    Ops/XpLinear.cpp
    Ops/XpMatmul.cpp
    Ops/XpRoPE.cpp
    Ops/XpSoftmax.cpp

    XnnpackBackend.cpp
    XpMemoryManager.cpp
)
target_include_directories(MllmXnnpackBackend PUBLIC third_party/XNNPACK/src/)
target_include_directories(MllmXnnpackBackend PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../)
target_link_libraries(MllmXnnpackBackend PUBLIC XNNPACK fmt::fmt-header-only fp16)
set_target_properties(MllmXnnpackBackend PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
