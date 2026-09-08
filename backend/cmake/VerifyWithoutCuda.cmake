# This target proves physical directory independence using a fresh source copy.
# It never moves or deletes the caller's CUDA files, and copies no build outputs.
foreach(required IN ITEMS NYA_SOURCE_ROOT NYA_VERIFY_PARENT NYA_GENERATOR NYA_COMPILER)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "Missing verification argument ${required}")
    endif()
endforeach()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(check_root "${NYA_VERIFY_PARENT}/without-cuda-${suffix}")
file(MAKE_DIRECTORY "${check_root}/source")
file(COPY "${NYA_SOURCE_ROOT}/CMakeLists.txt" "${NYA_SOURCE_ROOT}/config.yaml"
    DESTINATION "${check_root}/source")
file(COPY "${NYA_SOURCE_ROOT}/backend" DESTINATION "${check_root}/source"
    PATTERN cuda EXCLUDE PATTERN CUDA EXCLUDE PATTERN "build*" EXCLUDE)
if(EXISTS "${check_root}/source/backend/cuda" OR EXISTS "${check_root}/source/CUDA")
    message(FATAL_ERROR "Verification copy unexpectedly contains CUDA")
endif()
set(options -G "${NYA_GENERATOR}" "-DCMAKE_C_COMPILER=${NYA_COMPILER}"
    "-DCMAKE_BUILD_TYPE=Release" "-DNYA_C_STANDARD=${NYA_STANDARD}"
    -DNYA_ENABLE_ONNXRUNTIME=OFF -DNYA_WARNINGS_AS_ERRORS=ON -DNYA_ENABLE_CUDA=ON
    "-DNYA_ENABLE_VULKAN=${NYA_VULKAN}" -DBUILD_TESTING=ON)
if(NYA_MAKE_PROGRAM)
    list(APPEND options "-DCMAKE_MAKE_PROGRAM=${NYA_MAKE_PROGRAM}")
endif()
if(NYA_GENERATOR_PLATFORM)
    list(APPEND options -A "${NYA_GENERATOR_PLATFORM}")
endif()
if(NYA_VULKAN)
    list(APPEND options "-DVulkan_INCLUDE_DIR=${NYA_VULKAN_INCLUDE}" "-DVulkan_LIBRARY=${NYA_VULKAN_LIBRARY}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -S "${check_root}/source" -B "${check_root}/build" ${options}
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${check_root}/build" --config Release --parallel 4
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${check_root}/build" -C Release --output-on-failure
    COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "Verified CUDA-free source: ${check_root}/source (Vulkan=${NYA_VULKAN})")
