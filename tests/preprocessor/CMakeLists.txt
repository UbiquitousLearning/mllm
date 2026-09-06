add_executable(
  Mllm-Test-ChatTemplate
  ChatTemplateTest.cpp
  ${PROJECT_SOURCE_DIR}/mllm/preprocessor/chat_template/ChatTemplate.cpp)
target_link_libraries(Mllm-Test-ChatTemplate PRIVATE gtest_main)
target_include_directories(Mllm-Test-ChatTemplate PRIVATE ${MLLM_INCLUDE_DIR} ${MLLM_JSON_INCLUDE_DIR})
if(MLLM_ENABLE_JINJA_CHAT_TEMPLATE)
  target_link_libraries(Mllm-Test-ChatTemplate PRIVATE mllm_jinja_cpp)
  target_compile_definitions(Mllm-Test-ChatTemplate PRIVATE MLLM_ENABLE_JINJA_CHAT_TEMPLATE=1)
else()
  target_compile_definitions(Mllm-Test-ChatTemplate PRIVATE MLLM_ENABLE_JINJA_CHAT_TEMPLATE=0)
endif()

# Register the executable directly (not gtest_discover_tests) so Android
# cross-builds do not try to execute it on the build host.
add_test(NAME ChatTemplateFocused COMMAND Mllm-Test-ChatTemplate)

# Product-path probe used by compare_transformers_chat_template.py. It is
# built for both Jinja settings so the OFF build can prove that jinja_required
# fails closed.
add_executable(Mllm-ChatTemplate-Probe ChatTemplateProbe.cpp)
target_link_libraries(Mllm-ChatTemplate-Probe PRIVATE MllmCPUBackend)
target_include_directories(Mllm-ChatTemplate-Probe PRIVATE ${MLLM_INCLUDE_DIR} ${MLLM_JSON_INCLUDE_DIR})

if(MLLM_ENABLE_JINJA_CHAT_TEMPLATE)
  add_executable(
    Mllm-ChatTemplate-Render
    ChatTemplateRender.cpp
    ${PROJECT_SOURCE_DIR}/mllm/preprocessor/chat_template/ChatTemplate.cpp)
  target_link_libraries(Mllm-ChatTemplate-Render PRIVATE mllm_jinja_cpp)
  target_compile_definitions(Mllm-ChatTemplate-Render PRIVATE MLLM_ENABLE_JINJA_CHAT_TEMPLATE=1)
  target_include_directories(Mllm-ChatTemplate-Render PRIVATE ${MLLM_INCLUDE_DIR} ${MLLM_JSON_INCLUDE_DIR})
endif()
