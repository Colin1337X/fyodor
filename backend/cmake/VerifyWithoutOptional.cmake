# Copy only source directories into a unique verification root. Never move or
# delete anything in the user's checkout. Preserve every other provider.
if(NOT NYA_EXCLUDE MATCHES "^(rocm|mlx|optcpp)$")
    message(FATAL_ERROR "Invalid optional directory")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(root "${NYA_VERIFY_PARENT}/without-${NYA_EXCLUDE}-${suffix}")
file(MAKE_DIRECTORY "${root}/source")
file(COPY "${NYA_SOURCE_ROOT}/CMakeLists.txt" "${NYA_SOURCE_ROOT}/config.yaml" DESTINATION "${root}/source")
foreach(directory IN ITEMS backend CUDA rocm mlx)
    if(EXISTS "${NYA_SOURCE_ROOT}/${directory}" AND NOT directory STREQUAL NYA_EXCLUDE)
        file(COPY "${NYA_SOURCE_ROOT}/${directory}" DESTINATION "${root}/source"
            PATTERN "${NYA_EXCLUDE}" EXCLUDE PATTERN vendor EXCLUDE PATTERN "build*" EXCLUDE
            PATTERN benchmarks EXCLUDE)
    endif()
endforeach()
if(EXISTS "${root}/source/${NYA_EXCLUDE}" OR EXISTS "${root}/source/backend/${NYA_EXCLUDE}")
    message(FATAL_ERROR "Excluded directory exists in verification copy")
endif()
set(options -G "${NYA_GENERATOR}" "-DCMAKE_C_COMPILER=${NYA_COMPILER}"
    -DCMAKE_BUILD_TYPE=Release "-DNYA_C_STANDARD=${NYA_STANDARD}" -DNYA_WARNINGS_AS_ERRORS=ON
    -DNYA_ENABLE_ONNXRUNTIME=OFF -DBUILD_TESTING=ON -DNYA_ENABLE_ROCM=ON -DNYA_ENABLE_MLX=ON
    "-DNYA_ENABLE_CUDA=${NYA_CUDA}" "-DNYA_ENABLE_VULKAN=${NYA_VULKAN}"
    "-DNYA_CUDA_INCLUDE_DIR=${NYA_CUDA_INCLUDE}" "-DNYA_NVRTC_INCLUDE_DIR=${NYA_NVRTC_INCLUDE}"
    "-DNYA_CUDA_NVRTC_LIBRARY=${NYA_NVRTC_LIBRARY}"
    "-DVulkan_INCLUDE_DIR=${NYA_VULKAN_INCLUDE}" "-DVulkan_LIBRARY=${NYA_VULKAN_LIBRARY}")
if(NYA_MAKE_PROGRAM)
    list(APPEND options "-DCMAKE_MAKE_PROGRAM=${NYA_MAKE_PROGRAM}")
endif()
if(NYA_GENERATOR_PLATFORM)
    list(APPEND options -A "${NYA_GENERATOR_PLATFORM}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${root}/source" -B "${root}/build" ${options} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${root}/build" --config Release --parallel 4 COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${root}/build" -C Release --output-on-failure COMMAND_ERROR_IS_FATAL ANY)
# Use the same model and measurements for both binaries. The default synthetic
# decoder is a smoke benchmark; set NYA_VERIFY_MODEL to a real GGUF for useful
# performance comparisons. This target makes no significance claim from it.
if(NOT NYA_VERIFY_MODEL)
    set(NYA_VERIFY_MODEL "${root}/build/backend/test_models/pretraining.trained.gguf")
endif()
set(binary "${root}/build/fyodor-bench${NYA_SUFFIX}")
if(NOT EXISTS "${binary}")
    set(binary "${root}/build/Release/fyodor-bench${NYA_SUFFIX}")
endif()
foreach(version IN ITEMS before after)
    set(exe "${binary}")
    if(version STREQUAL "before")
        set(exe "${NYA_BASELINE_BENCH}")
    endif()
    execute_process(COMMAND "${exe}" -m "${NYA_VERIFY_MODEL}" -b cpu -p 8 -n 8 -r 5 --warmup 2 --json
        OUTPUT_FILE "${root}/${version}.json" COMMAND_ERROR_IS_FATAL ANY)
    file(READ "${root}/${version}.json" data)
    foreach(i RANGE 0 1)
        string(JSON ${version}_${i} GET "${data}" results ${i} tokens_per_second)
    endforeach()
endforeach()
message(STATUS "CPU pp8 before=${before_0}, after=${after_0}; tg8 before=${before_1}, after=${after_1} tokens/s")
message(STATUS "Verified without ${NYA_EXCLUDE}: ${root}; benchmark JSON retained")
