cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(fyodor VERSION 0.3.0 LANGUAGES C)
    include(CTest)
endif()

set(NYA_C_STANDARD 17 CACHE STRING "ISO C language version (17 or 23)")
set_property(CACHE NYA_C_STANDARD PROPERTY STRINGS 17 23)
if(NOT NYA_C_STANDARD MATCHES "^(17|23)$")
    message(FATAL_ERROR "NYA_C_STANDARD must be 17 or 23")
endif()
option(NYA_ENABLE_VULKAN "Enable independent Vulkan compute acceleration" OFF)
option(NYA_ENABLE_CUDA "Enable optional CUDA driver/NVRTC provider when its directory and headers exist" OFF)
option(NYA_ENABLE_ONNXRUNTIME "Enable the optional external ONNX Runtime C API adapter" OFF)
option(NYA_FETCH_ONNXRUNTIME "Download the pinned optional ONNX Runtime SDK" OFF)
option(NYA_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
option(NYA_ENABLE_IPO "Enable checked interprocedural optimization" OFF)
option(NYA_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build configuration" FORCE)
endif()
find_package(Threads REQUIRED)
if(NYA_ENABLE_SANITIZERS)
    # Probe the actual compiler/runtime pair instead of assuming that every
    # Windows toolchain lacks UBSan. LLVM-MinGW supports both sanitizers; some
    # other MinGW/MSVC distributions ship neither or only AddressSanitizer.
    include(CheckCSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=address,undefined -fno-sanitize-recover=all")
    check_c_source_compiles("int main(void) { return 0; }" NYA_HAS_SANITIZER_RUNTIME)
    cmake_pop_check_state()
    if(NOT NYA_HAS_SANITIZER_RUNTIME)
        message(FATAL_ERROR "This C compiler cannot link AddressSanitizer + UBSan; select a toolchain with both runtimes")
    endif()
endif()
if(NYA_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT nya_ipo_supported OUTPUT nya_ipo_error LANGUAGES C)
    if(NOT nya_ipo_supported)
        message(FATAL_ERROR "IPO is unavailable: ${nya_ipo_error}")
    endif()
endif()

# One policy covers production and tests. No fast-math or machine-specific
# instructions: distributed binaries remain portable and retain NaN checks.
function(nya_target_policy target)
    set_target_properties(${target} PROPERTIES
        C_STANDARD ${NYA_C_STANDARD} C_STANDARD_REQUIRED YES C_EXTENSIONS NO
        INTERPROCEDURAL_OPTIMIZATION ${NYA_ENABLE_IPO}
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    if(MSVC)
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
        target_compile_options(${target} PRIVATE /W4)
        if(NYA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(NYA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
    if(NOT WIN32)
        target_compile_definitions(${target} PRIVATE _POSIX_C_SOURCE=200809L _FILE_OFFSET_BITS=64)
    endif()
    if(NYA_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer)
        target_link_options(${target} PUBLIC -fsanitize=address,undefined -fno-sanitize-recover=all)
    endif()
endfunction()

# Compile the model engine once, independently from HTTP. The server and tests
# reuse the same objects, so neither can drift into a different implementation.
add_library(nya-engine STATIC
    core/compute.c core/execution.c core/generation.c core/file.c core/format.c
    core/format_gguf.c core/format_onnx.c core/format_safetensors.c
    core/llm_cpu.c core/llm_gguf.c core/multimodal.c core/model.c core/runtime.c core/training.c core/pretraining.c)
target_include_directories(nya-engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
nya_target_policy(nya-engine)
if(WIN32)
    target_link_libraries(nya-engine PUBLIC bcrypt)
else()
    target_link_libraries(nya-engine PUBLIC m ${CMAKE_DL_LIBS})
endif()
if(NYA_ENABLE_VULKAN)
    # Vulkan has its own dependency and source list; CUDA is never probed.
    find_package(Vulkan 1.3 REQUIRED)
    target_sources(nya-engine PRIVATE vulkan/vulkan.c)
    target_compile_definitions(nya-engine PRIVATE NYA_ENABLE_VULKAN=1)
    target_link_libraries(nya-engine PUBLIC Vulkan::Vulkan)
endif()
if(NYA_ENABLE_ONNXRUNTIME)
    include(cmake/OnnxRuntime.cmake)
endif()
if(NYA_ENABLE_CUDA)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CUDA/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../CUDA" "${CMAKE_CURRENT_BINARY_DIR}/cuda-provider")
    else()
        message(STATUS "CUDA directory absent: keeping CPU/Vulkan build")
    endif()
endif()

add_library(nya-server STATIC core/config.c core/server.c core/thread.c)
nya_target_policy(nya-server)
target_link_libraries(nya-server PUBLIC nya-engine Threads::Threads)
if(WIN32)
    target_link_libraries(nya-server PUBLIC ws2_32)
endif()
add_executable(fyodor-backend core/main.c)
nya_target_policy(fyodor-backend)
target_link_libraries(fyodor-backend PRIVATE nya-server)
add_executable(fyodor-train core/train_main.c)
nya_target_policy(fyodor-train)
target_link_libraries(fyodor-train PRIVATE nya-engine)
function(nya_copy_provider target)
    if(NYA_ENABLE_ONNXRUNTIME)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${NYA_ONNXRUNTIME_SHARED_LIBRARY}" "$<TARGET_FILE_DIR:${target}>"
            VERBATIM)
    endif()
endfunction()
nya_copy_provider(fyodor-backend)
nya_copy_provider(fyodor-train)

add_custom_target(verify-without-cuda
    COMMAND "${CMAKE_COMMAND}"
        "-DNYA_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
        "-DNYA_VERIFY_PARENT=${CMAKE_BINARY_DIR}"
        "-DNYA_GENERATOR=${CMAKE_GENERATOR}"
        "-DNYA_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DNYA_COMPILER=${CMAKE_C_COMPILER}"
        "-DNYA_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DNYA_STANDARD=${NYA_C_STANDARD}"
        "-DNYA_VULKAN=${NYA_ENABLE_VULKAN}"
        "-DNYA_VULKAN_INCLUDE=${Vulkan_INCLUDE_DIR}"
        "-DNYA_VULKAN_LIBRARY=${Vulkan_LIBRARY}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyWithoutCuda.cmake"
    USES_TERMINAL VERBATIM)

if(BUILD_TESTING)
    set(nya_fixtures "${CMAKE_CURRENT_BINARY_DIR}/test_models")
    file(MAKE_DIRECTORY "${nya_fixtures}")
    add_executable(nya-make-test-models tests/make_test_models.c)
    add_executable(nya-test-config tests/test_config.c core/config.c)
    add_executable(nya-test-formats tests/test_formats.c)
    add_executable(nya-test-execution tests/test_execution.c)
    add_executable(nya-test-generation tests/test_generation.c)
    add_executable(nya-test-architectures tests/test_architectures.c)
    add_executable(nya-test-training tests/test_training.c)
    add_executable(nya-test-pretraining tests/test_pretraining.c)
    add_executable(nya-test-compute tests/test_compute.c)
    # Parser tests include server.c to exercise private functions directly.
    add_executable(nya-test-server tests/test_server.c core/config.c core/thread.c)
    add_executable(nya-test-api tests/test_api.c core/config.c core/thread.c)
    foreach(target IN ITEMS nya-make-test-models nya-test-config nya-test-formats
            nya-test-execution nya-test-generation nya-test-architectures nya-test-training nya-test-pretraining nya-test-compute nya-test-server nya-test-api)
        nya_target_policy(${target})
    endforeach()
    foreach(target IN ITEMS nya-test-formats nya-test-execution nya-test-generation
            nya-test-architectures nya-test-training nya-test-pretraining nya-test-compute nya-test-server nya-test-api)
        target_link_libraries(${target} PRIVATE nya-engine)
        target_include_directories(${target} PRIVATE core)
    endforeach()
    target_link_libraries(nya-test-server PRIVATE Threads::Threads)
    target_link_libraries(nya-test-api PRIVATE Threads::Threads)
    if(WIN32)
        target_link_libraries(nya-test-server PRIVATE ws2_32)
        target_link_libraries(nya-test-api PRIVATE ws2_32)
    endif()
    nya_copy_provider(nya-test-execution)
    add_test(NAME generate-model-fixtures COMMAND nya-make-test-models "${nya_fixtures}")
    set_tests_properties(generate-model-fixtures PROPERTIES FIXTURES_SETUP model_fixtures)
    add_test(NAME validate-server-config COMMAND nya-test-config
        "${CMAKE_CURRENT_SOURCE_DIR}/../config.yaml" "${nya_fixtures}/invalid-config.yaml")
    add_test(NAME inspect-model-formats COMMAND nya-test-formats
        "${nya_fixtures}/minimal.gguf" "${nya_fixtures}/minimal.safetensors"
        "${nya_fixtures}/identity.onnx" "${nya_fixtures}/invalid.gguf"
        "${nya_fixtures}/invalid.safetensors" "${nya_fixtures}/invalid.onnx")
    set_tests_properties(inspect-model-formats PROPERTIES FIXTURES_REQUIRED model_fixtures)
    add_test(NAME generate-native-gguf COMMAND nya-test-generation "${nya_fixtures}/tiny-llama.gguf")
    add_test(NAME validate-gemma-graphs COMMAND nya-test-architectures "${nya_fixtures}/tiny-gemma4.gguf")
    add_test(NAME validate-training COMMAND nya-test-training)
    add_test(NAME train-export-decoder COMMAND nya-test-pretraining "${nya_fixtures}/pretraining")
    set_tests_properties(train-export-decoder PROPERTIES TIMEOUT 120)
    set_tests_properties(train-export-decoder PROPERTIES FIXTURES_SETUP trained_model)
    add_test(NAME compatible-rest-apis COMMAND nya-test-api "${nya_fixtures}/pretraining.trained.gguf")
    set_tests_properties(compatible-rest-apis PROPERTIES FIXTURES_REQUIRED trained_model TIMEOUT 60)
    add_test(NAME train-cli-workflows COMMAND "${CMAKE_COMMAND}" "-DTRAINER=$<TARGET_FILE:fyodor-train>"
        "-DFIXTURES=${nya_fixtures}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/TrainCli.cmake")
    set_tests_properties(train-cli-workflows PROPERTIES TIMEOUT 120)
    add_test(NAME validate-http-protocol COMMAND nya-test-server)
    add_test(NAME validate-compute COMMAND nya-test-compute)
    if(NYA_ENABLE_VULKAN)
        add_test(NAME validate-vulkan-compute COMMAND nya-test-compute --vulkan)
        set_tests_properties(validate-vulkan-compute PROPERTIES
            ENVIRONMENT "NYA_COMPUTE=vulkan" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME generate-native-gguf-vulkan COMMAND nya-test-generation "${nya_fixtures}/tiny-llama-vulkan.gguf")
        set_tests_properties(generate-native-gguf-vulkan PROPERTIES ENVIRONMENT "NYA_COMPUTE=vulkan" TIMEOUT 60)
    endif()
    if(NYA_CUDA_AVAILABLE)
        add_test(NAME validate-cuda-compute COMMAND nya-test-compute --cuda)
        set_tests_properties(validate-cuda-compute PROPERTIES
            ENVIRONMENT "NYA_COMPUTE=cuda" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME validate-cuda-gemma COMMAND nya-test-architectures "${nya_fixtures}/tiny-gemma4-cuda.gguf")
        set_tests_properties(validate-cuda-gemma PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda" TIMEOUT 120)
        add_test(NAME generate-native-gguf-cuda COMMAND nya-test-generation "${nya_fixtures}/tiny-llama-cuda.gguf")
        set_tests_properties(generate-native-gguf-cuda PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda" TIMEOUT 60)
    endif()
    set_tests_properties(generate-native-gguf validate-http-protocol validate-compute PROPERTIES TIMEOUT 60)
    if(NYA_ENABLE_ONNXRUNTIME)
        add_test(NAME execute-onnx-model COMMAND nya-test-execution "${nya_fixtures}/identity.onnx")
        set_tests_properties(execute-onnx-model PROPERTIES FIXTURES_REQUIRED model_fixtures)
    endif()
endif()
