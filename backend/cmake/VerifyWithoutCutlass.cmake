# Build/test an actual source tree containing CUDA but no cutlass directory.
# The user's live checkout and any installed CUTLASS sources are never moved.
foreach(required IN ITEMS NYA_SOURCE_ROOT NYA_VERIFY_PARENT NYA_GENERATOR NYA_COMPILER NYA_CUDA_INCLUDE NYA_NVRTC_INCLUDE)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing verification argument ${required}")
    endif()
endforeach()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(check_root "${NYA_VERIFY_PARENT}/without-cutlass-${suffix}")
file(MAKE_DIRECTORY "${check_root}/source")
file(COPY "${NYA_SOURCE_ROOT}/CMakeLists.txt" "${NYA_SOURCE_ROOT}/config.yaml" DESTINATION "${check_root}/source")
file(COPY "${NYA_SOURCE_ROOT}/backend" "${NYA_SOURCE_ROOT}/CUDA" DESTINATION "${check_root}/source"
    PATTERN cutlass EXCLUDE PATTERN vendor EXCLUDE PATTERN "build*" EXCLUDE)
if(EXISTS "${check_root}/source/CUDA/cutlass" OR NOT EXISTS "${check_root}/source/CUDA/cuda.c")
    message(FATAL_ERROR "Verification source does not have CUDA-without-CUTLASS layout")
endif()
set(options -G "${NYA_GENERATOR}" "-DCMAKE_C_COMPILER=${NYA_COMPILER}"
    -DCMAKE_BUILD_TYPE=Release "-DNYA_C_STANDARD=${NYA_STANDARD}"
    -DNYA_ENABLE_ONNXRUNTIME=OFF -DNYA_WARNINGS_AS_ERRORS=ON
    -DNYA_ENABLE_CUDA=ON -DNYA_ENABLE_CUTLASS=ON -DBUILD_TESTING=ON
    "-DNYA_CUDA_INCLUDE_DIR=${NYA_CUDA_INCLUDE}" "-DNYA_NVRTC_INCLUDE_DIR=${NYA_NVRTC_INCLUDE}"
    "-DNYA_CUDA_NVRTC_LIBRARY=${NYA_NVRTC_LIBRARY}" "-DNYA_ENABLE_VULKAN=${NYA_VULKAN}")
if(NYA_MAKE_PROGRAM)
    list(APPEND options "-DCMAKE_MAKE_PROGRAM=${NYA_MAKE_PROGRAM}")
endif()
if(NYA_GENERATOR_PLATFORM)
    list(APPEND options -A "${NYA_GENERATOR_PLATFORM}")
endif()
if(NYA_VULKAN)
    list(APPEND options "-DVulkan_INCLUDE_DIR=${NYA_VULKAN_INCLUDE}" "-DVulkan_LIBRARY=${NYA_VULKAN_LIBRARY}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${check_root}/source" -B "${check_root}/build" ${options} COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${check_root}/build" --config Release --parallel 4 COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${check_root}/build" -C Release --output-on-failure COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "Verified CUDA source with CUTLASS physically absent: ${check_root}/source")
