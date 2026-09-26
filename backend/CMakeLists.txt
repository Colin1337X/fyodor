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
option(NYA_ENABLE_ROCM "Enable detachable ROCm C adapter (optional runtime libraries)" ON)
option(NYA_ENABLE_MLX "Enable detachable MLX C adapter (optional runtime library)" ON)
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

# One policy covers production and tests. No global fast-math or ISA flags:
# specialized CPU functions require runtime feature checks before dispatch.
# Distributed binaries retain the scalar path and NaN validation.
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
    core/cpu_backend.c core/cpu_kernels.c core/llm_quant.c core/thread.c core/vendor.c
    core/compute.c core/execution.c core/generation.c core/file.c core/format.c
    core/format_gguf.c core/format_onnx.c core/format_safetensors.c
    core/llm_cpu.c core/llm_gguf.c core/multimodal.c core/model.c core/runtime.c core/training.c core/train_executor.c core/pretraining.c)
target_include_directories(nya-engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(nya-engine PUBLIC Threads::Threads)
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
foreach(provider IN ITEMS rocm mlx)
    string(TOUPPER "${provider}" upper)
    if(NYA_ENABLE_${upper})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${provider}/CMakeLists.txt" AND
           EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${provider}/${provider}.c")
            add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../${provider}" "${CMAKE_CURRENT_BINARY_DIR}/${provider}-provider")
        else()
            message(STATUS "${provider} directory absent: retaining independent providers")
        endif()
    endif()
endforeach()
if(NYA_ENABLE_CUDA)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CUDA/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../CUDA" "${CMAKE_CURRENT_BINARY_DIR}/cuda-provider")
    else()
        message(STATUS "CUDA directory absent: keeping CPU/Vulkan build")
    endif()
endif()

add_library(nya-server STATIC core/config.c core/server.c)
nya_target_policy(nya-server)
target_link_libraries(nya-server PUBLIC nya-engine Threads::Threads)
if(WIN32)
    target_link_libraries(nya-server PUBLIC ws2_32)
endif()
add_executable(fyodor-backend core/main.c)
nya_target_policy(fyodor-backend)
target_link_libraries(fyodor-backend PRIVATE nya-server)
add_executable(fyodor-train core/train_main.c)
add_executable(fyodor-bench core/bench_main.c)
nya_target_policy(fyodor-bench)
target_link_libraries(fyodor-bench PRIVATE nya-engine)
nya_target_policy(fyodor-train)
target_link_libraries(fyodor-train PRIVATE nya-engine)
if(MINGW)
    # Use the UTF-16 CRT entry point, then convert arguments to internal UTF-8.
    target_link_options(fyodor-train PRIVATE -municode)
endif()
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

set(NYA_VERIFY_MODEL "" CACHE FILEPATH "Optional real GGUF for physical-removal benchmark comparison")
foreach(optional IN ITEMS rocm mlx optcpp)
    add_custom_target(verify-without-${optional}
        COMMAND "${CMAKE_COMMAND}"
            "-DNYA_EXCLUDE=${optional}"
            "-DNYA_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
            "-DNYA_VERIFY_PARENT=${CMAKE_BINARY_DIR}"
            "-DNYA_GENERATOR=${CMAKE_GENERATOR}"
            "-DNYA_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DNYA_COMPILER=${CMAKE_C_COMPILER}"
            "-DNYA_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DNYA_STANDARD=${NYA_C_STANDARD}"
            "-DNYA_CUDA=${NYA_ENABLE_CUDA}"
            "-DNYA_CUDA_INCLUDE=${NYA_CUDA_INCLUDE_DIR}"
            "-DNYA_NVRTC_INCLUDE=${NYA_NVRTC_INCLUDE_DIR}"
            "-DNYA_NVRTC_LIBRARY=${NYA_CUDA_NVRTC_LIBRARY}"
            "-DNYA_VULKAN=${NYA_ENABLE_VULKAN}"
            "-DNYA_VULKAN_INCLUDE=${Vulkan_INCLUDE_DIR}"
            "-DNYA_VULKAN_LIBRARY=${Vulkan_LIBRARY}"
            "-DNYA_BASELINE_BENCH=$<TARGET_FILE:fyodor-bench>"
            "-DNYA_VERIFY_MODEL=${NYA_VERIFY_MODEL}"
            "-DNYA_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyWithoutOptional.cmake"
        DEPENDS fyodor-bench USES_TERMINAL VERBATIM)
endforeach()

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

if(NYA_CUDA_AVAILABLE)
    add_custom_target(verify-without-cutlass
        COMMAND "${CMAKE_COMMAND}"
            "-DNYA_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/.."
            "-DNYA_VERIFY_PARENT=${CMAKE_BINARY_DIR}"
            "-DNYA_GENERATOR=${CMAKE_GENERATOR}"
            "-DNYA_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DNYA_COMPILER=${CMAKE_C_COMPILER}"
            "-DNYA_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            "-DNYA_STANDARD=${NYA_C_STANDARD}"
            "-DNYA_CUDA_INCLUDE=${NYA_CUDA_INCLUDE_DIR}"
            "-DNYA_NVRTC_INCLUDE=${NYA_NVRTC_INCLUDE_DIR}"
            "-DNYA_NVRTC_LIBRARY=${NYA_CUDA_NVRTC_LIBRARY}"
            "-DNYA_VULKAN=${NYA_ENABLE_VULKAN}"
            "-DNYA_VULKAN_INCLUDE=${Vulkan_INCLUDE_DIR}"
            "-DNYA_VULKAN_LIBRARY=${Vulkan_LIBRARY}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyWithoutCutlass.cmake"
        USES_TERMINAL VERBATIM)
endif()

if(BUILD_TESTING)
    if(NYA_ROCM_AVAILABLE OR NYA_MLX_AVAILABLE)
        # Explicit test library path only; this is never installed alongside
        # the application and cannot masquerade as a production GPU runtime.
        add_library(nya-mock-vendor SHARED tests/mock_vendor.c)
        nya_target_policy(nya-mock-vendor)
        add_library(nya-mock-missing-vendor SHARED tests/mock_vendor.c)
        nya_target_policy(nya-mock-missing-vendor)
        target_compile_definitions(nya-mock-missing-vendor PRIVATE hipInit=nya_mock_unused_init mlx_set_error_handler=nya_mock_unused_handler)
        add_executable(nya-test-vendor tests/test_vendor.c)
        nya_target_policy(nya-test-vendor)
        target_include_directories(nya-test-vendor PRIVATE core)
        target_link_libraries(nya-test-vendor PRIVATE nya-engine)
        foreach(provider IN ITEMS rocm mlx)
            string(TOUPPER "${provider}" upper)
            if(NYA_${upper}_AVAILABLE)
                set(mock_env "NYA_COMPUTE=${provider};NYA_ROCM_HIP_LIBRARY=$<TARGET_FILE:nya-mock-vendor>;NYA_ROCM_BLAS_LIBRARY=$<TARGET_FILE:nya-mock-vendor>;NYA_MLX_LIBRARY=$<TARGET_FILE:nya-mock-vendor>")
                add_test(NAME vendor-${provider}-failures COMMAND nya-test-vendor "${provider}" "$<TARGET_FILE:nya-mock-vendor>")
                set_tests_properties(vendor-${provider}-failures PROPERTIES ENVIRONMENT "${mock_env};NYA_ROCM_CACHE_MIB=1;NYA_MLX_CACHE_MIB=1" TIMEOUT 60)
                add_test(NAME vendor-${provider}-quant COMMAND nya-test-quant-kernels)
                set_tests_properties(vendor-${provider}-quant PROPERTIES ENVIRONMENT "${mock_env}" TIMEOUT 120)
                add_test(NAME vendor-${provider}-inference COMMAND nya-test-inference "${CMAKE_CURRENT_BINARY_DIR}/test_models/pretraining.trained.gguf" gpu-assisted)
                set_tests_properties(vendor-${provider}-inference PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "${mock_env}" TIMEOUT 120)
                add_test(NAME vendor-${provider}-gemma COMMAND nya-test-architectures "${CMAKE_CURRENT_BINARY_DIR}/test_models/tiny-gemma4-${provider}.gguf")
                set_tests_properties(vendor-${provider}-gemma PROPERTIES ENVIRONMENT "${mock_env}" TIMEOUT 120)
                add_test(NAME vendor-${provider}-api COMMAND nya-test-api "${CMAKE_CURRENT_BINARY_DIR}/test_models/pretraining.trained.gguf")
                set_tests_properties(vendor-${provider}-api PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "${mock_env};NYA_COMPUTE=cpu;NYA_TEST_OPTIONAL_API=${provider}" TIMEOUT 120)
                foreach(failure IN ITEMS missing-library missing-symbol invalid-budget)
                    set(failure_env "${mock_env}")
                    if(failure STREQUAL "invalid-budget")
                        string(APPEND failure_env ";NYA_ROCM_CACHE_MIB=-1;NYA_MLX_CACHE_MIB=-1")
                    else()
                        set(library "${CMAKE_CURRENT_BINARY_DIR}/nonexistent-vendor-library")
                        if(failure STREQUAL "missing-symbol")
                            set(library "$<TARGET_FILE:nya-mock-missing-vendor>")
                        endif()
                        set(failure_env "NYA_COMPUTE=${provider};NYA_ROCM_HIP_LIBRARY=${library};NYA_ROCM_BLAS_LIBRARY=${library};NYA_MLX_LIBRARY=${library}")
                    endif()
                    add_test(NAME vendor-${provider}-${failure} COMMAND nya-test-vendor --unavailable)
                    set_tests_properties(vendor-${provider}-${failure} PROPERTIES ENVIRONMENT "${failure_env}" TIMEOUT 30)
                    if(failure STREQUAL "missing-symbol")
                        add_test(NAME vendor-${provider}-api-fallback COMMAND nya-test-api "${CMAKE_CURRENT_BINARY_DIR}/test_models/pretraining.trained.gguf")
                        set_tests_properties(vendor-${provider}-api-fallback PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "${failure_env};NYA_COMPUTE=cpu;NYA_TEST_OPTIONAL_API=${provider};NYA_TEST_OPTIONAL_UNAVAILABLE=1" TIMEOUT 120)
                    endif()
                endforeach()
                add_test(NAME hardware-${provider}-quant COMMAND nya-test-quant-kernels)
                set_tests_properties(hardware-${provider}-quant PROPERTIES ENVIRONMENT "NYA_COMPUTE=${provider}" SKIP_RETURN_CODE 77 TIMEOUT 120)
                add_test(NAME hardware-${provider}-inference COMMAND nya-test-inference "${CMAKE_CURRENT_BINARY_DIR}/test_models/pretraining.trained.gguf" gpu-assisted)
                set_tests_properties(hardware-${provider}-inference PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=${provider}" SKIP_RETURN_CODE 77 TIMEOUT 120)
                add_test(NAME vendor-${provider}-bench COMMAND "${CMAKE_COMMAND}" "-DBENCH=$<TARGET_FILE:fyodor-bench>"
                    "-DMODEL=${CMAKE_CURRENT_BINARY_DIR}/test_models/pretraining.trained.gguf" "-DPROVIDER=${provider}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/BenchVendor.cmake")
                set_tests_properties(vendor-${provider}-bench PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "${mock_env}" TIMEOUT 120)
            endif()
        endforeach()
    endif()
    set(nya_fixtures "${CMAKE_CURRENT_BINARY_DIR}/test_models")
    file(MAKE_DIRECTORY "${nya_fixtures}")
    add_executable(nya-make-test-models tests/make_test_models.c)
    add_executable(nya-test-config tests/test_config.c core/config.c)
    add_executable(nya-test-formats tests/test_formats.c)
    add_executable(nya-test-execution tests/test_execution.c)
    add_executable(nya-test-generation tests/test_generation.c)
    add_executable(nya-test-architectures tests/test_architectures.c)
    add_executable(nya-test-training tests/test_training.c)
    add_executable(nya-test-train-runner tests/test_train_runner.c)
    add_executable(nya-test-pretraining tests/test_pretraining.c)
    add_executable(nya-test-compute tests/test_compute.c)
    add_executable(nya-test-inference tests/test_inference.c)
    add_executable(nya-test-quant-kernels tests/test_quant_kernels.c)
    add_executable(nya-test-attention tests/test_attention.c)
    add_executable(nya-test-kv-precision tests/test_kv_precision.c)
    nya_target_policy(nya-test-kv-precision)
    target_link_libraries(nya-test-kv-precision PRIVATE nya-engine)
    target_include_directories(nya-test-kv-precision PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/core")
    add_test(NAME kv-precision COMMAND nya-test-kv-precision)
    add_test(NAME quant-kernels-cpu-dot COMMAND nya-test-quant-kernels)
    set_tests_properties(quant-kernels-cpu-dot PROPERTIES ENVIRONMENT "NYA_COMPUTE=cpu;NYA_CPU_GEMM=dot" TIMEOUT 60)
    nya_target_policy(nya-test-attention)
    target_include_directories(nya-test-attention PRIVATE core)
    target_link_libraries(nya-test-attention PRIVATE nya-engine)
    nya_target_policy(nya-test-quant-kernels)
    target_include_directories(nya-test-quant-kernels PRIVATE core)
    target_link_libraries(nya-test-quant-kernels PRIVATE nya-engine)
    foreach(isa IN ITEMS scalar avx2 avx512)
        add_test(NAME quant-kernels-${isa} COMMAND nya-test-quant-kernels)
        add_test(NAME attention-${isa} COMMAND nya-test-attention)
        set_tests_properties(attention-${isa} PROPERTIES ENVIRONMENT "NYA_CPU_ISA=${isa}" TIMEOUT 60)
        set_tests_properties(quant-kernels-${isa} PROPERTIES ENVIRONMENT "NYA_COMPUTE=cpu;NYA_CPU_ISA=${isa}" TIMEOUT 60)
    endforeach()
    nya_target_policy(nya-test-inference)
    target_include_directories(nya-test-inference PRIVATE core)
    target_link_libraries(nya-test-inference PRIVATE nya-engine)
    add_test(NAME inference-session-parity COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" cpu-reference)
    set_tests_properties(inference-session-parity PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_INFERENCE_REFERENCE=1")
    add_test(NAME inference-cpu-optimized COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" cpu-optimized)
    set_tests_properties(inference-cpu-optimized PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cpu")
    # Parser tests include server.c to exercise private functions directly.
    add_executable(nya-test-server tests/test_server.c core/config.c)
    add_executable(nya-test-api tests/test_api.c core/config.c)
    foreach(target IN ITEMS nya-make-test-models nya-test-config nya-test-formats
            nya-test-execution nya-test-generation nya-test-architectures nya-test-training nya-test-train-runner nya-test-pretraining nya-test-compute nya-test-server nya-test-api)
        nya_target_policy(${target})
    endforeach()
    foreach(target IN ITEMS nya-test-formats nya-test-execution nya-test-generation
            nya-test-architectures nya-test-training nya-test-train-runner nya-test-pretraining nya-test-compute nya-test-server nya-test-api)
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
    add_test(NAME train-accumulation COMMAND nya-test-train-runner)
    add_test(NAME validate-training-scalar COMMAND nya-test-training)
    set_tests_properties(validate-training-scalar PROPERTIES ENVIRONMENT "NYA_CPU_ISA=scalar")
    add_test(NAME train-export-decoder COMMAND nya-test-pretraining "${nya_fixtures}/pretraining")
    set_tests_properties(train-export-decoder PROPERTIES TIMEOUT 120)
    set_tests_properties(train-export-decoder PROPERTIES FIXTURES_SETUP trained_model)
    add_test(NAME compatible-rest-apis COMMAND nya-test-api "${nya_fixtures}/pretraining.trained.gguf")
    set_tests_properties(compatible-rest-apis PROPERTIES FIXTURES_REQUIRED trained_model TIMEOUT 60)
    add_test(NAME benchmark-cli COMMAND "${CMAKE_COMMAND}" "-DBENCH=$<TARGET_FILE:fyodor-bench>"
        "-DMODEL=${nya_fixtures}/pretraining.trained.gguf" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/BenchCli.cmake")
    set_tests_properties(benchmark-cli PROPERTIES FIXTURES_REQUIRED trained_model TIMEOUT 60)
    add_test(NAME train-cli-workflows COMMAND "${CMAKE_COMMAND}" "-DTRAINER=$<TARGET_FILE:fyodor-train>"
        "-DFIXTURES=${nya_fixtures}" -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/TrainCli.cmake")
    set_tests_properties(train-cli-workflows PROPERTIES TIMEOUT 120)
    # Python drives real child-process interruption; never a runtime dependency.
    # Native control-pipe checks also run in train-accumulation without Python.
    find_package(Python3 QUIET COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME train-stop-workflows COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/train_stop.py" "$<TARGET_FILE:fyodor-train>")
        set_tests_properties(train-stop-workflows PROPERTIES TIMEOUT 120)
        add_test(NAME train-unicode-paths COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/train_paths.py" "$<TARGET_FILE:fyodor-train>")
        set_tests_properties(train-unicode-paths PROPERTIES TIMEOUT 120)
        add_test(NAME train-evaluation COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/train_eval.py" "$<TARGET_FILE:fyodor-train>")
        set_tests_properties(train-evaluation PROPERTIES TIMEOUT 120)
    endif()
    add_test(NAME validate-http-protocol COMMAND nya-test-server)
    add_test(NAME validate-compute COMMAND nya-test-compute)
    if(NYA_ENABLE_VULKAN)
        add_test(NAME validate-vulkan-compute COMMAND nya-test-compute --vulkan)
        set_tests_properties(validate-vulkan-compute PROPERTIES
            ENVIRONMENT "NYA_COMPUTE=vulkan" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME generate-native-gguf-vulkan COMMAND nya-test-generation "${nya_fixtures}/tiny-llama-vulkan.gguf")
        set_tests_properties(generate-native-gguf-vulkan PROPERTIES ENVIRONMENT "NYA_COMPUTE=vulkan" TIMEOUT 60)
        add_test(NAME inference-vulkan-logits COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" gpu-matvec)
        set_tests_properties(inference-vulkan-logits PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=vulkan" SKIP_RETURN_CODE 77 TIMEOUT 60)
    endif()
    if(NYA_CUDA_AVAILABLE)
        add_test(NAME inference-cuda-attention-shapes COMMAND nya-test-inference --attention-shapes "${nya_fixtures}/attention-shape")
        set_tests_properties(inference-cuda-attention-shapes PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_GRAPH=1;NYA_TEST_ATTENTION_TILES=1" SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME inference-cuda-attention-unsplit COMMAND nya-test-inference --attention-shapes "${nya_fixtures}/attention-unsplit")
        set_tests_properties(inference-cuda-attention-unsplit PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_GRAPH=1;NYA_TEST_ATTENTION_TILES=1;NYA_CUDA_SPLIT_ATTENTION=0" SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME inference-cuda-attention-no-graphs COMMAND nya-test-inference --attention-shapes "${nya_fixtures}/attention-no-graphs")
        set_tests_properties(inference-cuda-attention-no-graphs PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_GRAPH=1;NYA_TEST_ATTENTION_TILES=1;NYA_CUDA_GRAPHS=0" SKIP_RETURN_CODE 77 TIMEOUT 180)
        add_test(NAME inference-cuda-graph-reuse COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-graph-reuse PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_GRAPH=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME inference-cuda-graph-reference COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-graph-reference PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_GRAPH=1;NYA_CUDA_REFERENCE=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME inference-cuda-blas COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-blas PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_BLAS=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME quant-kernels-cuda-chunks COMMAND nya-test-quant-kernels)
        set_tests_properties(quant-kernels-cuda-chunks PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_BLAS_CHUNKS=1;NYA_CUDA_BLAS_SCRATCH_MIB=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME quant-kernels-cuda-legacy-layout COMMAND nya-test-quant-kernels)
        set_tests_properties(quant-kernels-cuda-legacy-layout PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_CUTLASS=0;NYA_CUDA_BLAS_TRANSPOSE=0;NYA_TEST_BLAS_CHUNKS=1;NYA_CUDA_BLAS_SCRATCH_MIB=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME inference-cuda-blas-unavailable COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-blas-unavailable PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_CUTLASS=0;NYA_TEST_NO_BLAS=1;NYA_CUDA_BLAS_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/deliberately-missing-blas.dll" SKIP_RETURN_CODE 77 TIMEOUT 60)
        if(NYA_ENABLE_CUTLASS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../CUDA/cutlass/CMakeLists.txt")
            add_test(NAME inference-cuda-cutlass COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
            set_tests_properties(inference-cuda-cutlass PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_CUTLASS=1;NYA_CUDA_BLAS=0;NYA_TEST_CUTLASS=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
            add_test(NAME inference-cuda-cutlass-auto COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
            set_tests_properties(inference-cuda-cutlass-auto PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_BLAS=0;NYA_TEST_CUTLASS=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
            add_test(NAME inference-cuda-cutlass-disabled COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
            set_tests_properties(inference-cuda-cutlass-disabled PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_CUTLASS=0;NYA_CUDA_BLAS=0;NYA_TEST_NO_BLAS=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
            add_test(NAME quant-kernels-cuda-cutlass COMMAND nya-test-quant-kernels)
            set_tests_properties(quant-kernels-cuda-cutlass PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_CUTLASS=1;NYA_CUDA_BLAS=0;NYA_TEST_CUTLASS=1;NYA_TEST_BLAS_CHUNKS=1;NYA_CUDA_BLAS_SCRATCH_MIB=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
            add_test(NAME inference-cuda-cutlass-unavailable COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
            set_tests_properties(inference-cuda-cutlass-unavailable PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_CUTLASS=1;NYA_CUDA_CUTLASS_LIBRARY=${CMAKE_CURRENT_BINARY_DIR}/deliberately-missing-cutlass.dll;NYA_CUDA_BLAS=0;NYA_TEST_NO_BLAS=1" SKIP_RETURN_CODE 77 TIMEOUT 60)
        endif()
        add_test(NAME quant-kernels-cuda COMMAND nya-test-quant-kernels)
        set_tests_properties(quant-kernels-cuda PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda" TIMEOUT 60)
        add_test(NAME quant-kernels-cuda-reference COMMAND nya-test-quant-kernels)
        set_tests_properties(quant-kernels-cuda-reference PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_REFERENCE=1" TIMEOUT 60)
        add_test(NAME inference-cuda-resident COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-resident PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda" TIMEOUT 60)
        add_test(NAME inference-cuda-memory-fallback COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" cpu-reference)
        set_tests_properties(inference-cuda-memory-fallback PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_MEMORY_MIB=0" TIMEOUT 60)
        add_test(NAME inference-cuda-partial-allocation COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" cpu-reference)
        set_tests_properties(inference-cuda-partial-allocation PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_FAIL_ALLOCATION_AFTER=2" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME inference-cuda-prefix-replay COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-prefix-replay PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_FAIL_AFTER=3;NYA_TEST_DECODE=1" TIMEOUT 60)
        add_test(NAME inference-cuda-without-graphs COMMAND nya-test-inference "${nya_fixtures}/pretraining.trained.gguf" resident)
        set_tests_properties(inference-cuda-without-graphs PROPERTIES FIXTURES_REQUIRED trained_model ENVIRONMENT "NYA_COMPUTE=cuda;NYA_CUDA_GRAPHS=0;NYA_TEST_DECODE=1" TIMEOUT 60)
        add_test(NAME validate-cuda-compute COMMAND nya-test-compute --cuda)
        set_tests_properties(validate-cuda-compute PROPERTIES
            ENVIRONMENT "NYA_COMPUTE=cuda" SKIP_RETURN_CODE 77 TIMEOUT 60)
        add_test(NAME validate-cuda-gemma COMMAND nya-test-architectures "${nya_fixtures}/tiny-gemma4-cuda.gguf")
        set_tests_properties(validate-cuda-gemma PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda;NYA_TEST_GRAPH=1" TIMEOUT 120)
        add_test(NAME generate-native-gguf-cuda COMMAND nya-test-generation "${nya_fixtures}/tiny-llama-cuda.gguf")
        set_tests_properties(generate-native-gguf-cuda PROPERTIES ENVIRONMENT "NYA_COMPUTE=cuda" TIMEOUT 60)
        set_tests_properties(quant-kernels-cuda quant-kernels-cuda-reference inference-cuda-resident
            inference-cuda-memory-fallback inference-cuda-prefix-replay inference-cuda-without-graphs
            PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    set_tests_properties(generate-native-gguf validate-http-protocol validate-compute PROPERTIES TIMEOUT 60)
    if(NYA_ENABLE_ONNXRUNTIME)
        add_test(NAME execute-onnx-model COMMAND nya-test-execution "${nya_fixtures}/identity.onnx")
        set_tests_properties(execute-onnx-model PROPERTIES FIXTURES_REQUIRED model_fixtures)
    endif()
endif()
