option(GGML_CARTESI "ggml: use Cartesi software floating point" OFF)

if (GGML_CARTESI)
    ggml_add_backend_library(ggml-cartesi
        ggml-cartesi.cpp
        ggml-cartesi.h)

    target_include_directories(ggml-cartesi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_definitions(ggml-cartesi PRIVATE GGML_USE_CARTESI)
endif() 