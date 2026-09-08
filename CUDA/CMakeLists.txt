# This directory owns all CUDA/NVRTC dependencies and implementation. The parent
# only enters it when present; deleting it cannot leave required source paths.
find_path(NYA_CUDA_INCLUDE_DIR cuda.h HINTS "$ENV{CUDA_PATH}/include" "$ENV{CUDA_HOME}/include")
find_path(NYA_NVRTC_INCLUDE_DIR nvrtc.h HINTS "${NYA_CUDA_INCLUDE_DIR}" "$ENV{CUDA_PATH}/include" "$ENV{CUDA_HOME}/include")
set(NYA_CUDA_NVRTC_LIBRARY "" CACHE FILEPATH "Optional absolute path to the NVRTC runtime library")
if(NOT NYA_CUDA_INCLUDE_DIR OR NOT NYA_NVRTC_INCLUDE_DIR)
    message(STATUS "CUDA headers unavailable: keeping CPU/Vulkan build (set NYA_CUDA_INCLUDE_DIR and NYA_NVRTC_INCLUDE_DIR)")
    return()
endif()
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR (NOT WIN32 AND NOT UNIX))
    message(STATUS "CUDA dynamic loader is unavailable on this platform")
    return()
endif()

# Embed the exact CUDA source as bytes: no nvcc, C++ host compiler or shader
# generation tool is needed. NVRTC compiles only this source at runtime.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/matvec.cu")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/matvec.cu" nya_cuda_hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," nya_cuda_bytes "${nya_cuda_hex}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_source.h" "static const unsigned char nya_cuda_source[] = {${nya_cuda_bytes}0};\n")
file(TO_CMAKE_PATH "${NYA_CUDA_NVRTC_LIBRARY}" nya_nvrtc_default)
string(REPLACE "\"" "\\\"" nya_nvrtc_default "${nya_nvrtc_default}")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/cuda_source.h" "static const char nya_nvrtc_default[] = \"${nya_nvrtc_default}\";\n")
target_sources(nya-engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cuda.c")
target_include_directories(nya-engine PRIVATE "${NYA_CUDA_INCLUDE_DIR}" "${NYA_NVRTC_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(nya-engine PRIVATE NYA_ENABLE_CUDA=1)
set(NYA_CUDA_AVAILABLE TRUE PARENT_SCOPE)
message(STATUS "CUDA driver/NVRTC provider enabled (C host compiler; runtime CUDA compilation)")
