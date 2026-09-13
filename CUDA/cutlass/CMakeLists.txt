cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(fyodor-cutlass LANGUAGES NONE)
endif()

# C hosts may load a prebuilt bridge even without an installed C++/CUDA compiler.
# The include is guarded at its only call site, so removing this whole directory
# also removes all compile-time references to the optional interface.
if(TARGET nya-engine)
    target_compile_definitions(nya-engine PRIVATE NYA_ENABLE_CUTLASS=1)
endif()
option(NYA_BUILD_CUTLASS "Build the optional CUTLASS C ABI library when dependencies exist" ON)
set(NYA_CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/vendor/cutlass" CACHE PATH "CUTLASS source (validated v4.5.1)")
if(NOT NYA_BUILD_CUTLASS)
    return()
endif()
if(NOT EXISTS "${NYA_CUTLASS_ROOT}/include/cutlass/gemm/device/gemm.h")
    message(STATUS "CUTLASS headers absent: retaining native CUDA; see CUDA/cutlass/README.md")
    return()
endif()
if(WIN32 AND CMAKE_GENERATOR MATCHES "MinGW|MSYS")
    message(STATUS "CUTLASS: this generator cannot compile CUDA C++; use a standalone CUDA/cutlass build. C/NVRTC build retained")
    return()
endif()
include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(STATUS "CUTLASS CUDA compiler unavailable: retaining native CUDA")
    return()
endif()
enable_language(CUDA)
add_library(fyodor-cutlass SHARED gemm.cu)
target_compile_definitions(fyodor-cutlass PRIVATE NYA_CUTLASS_IMPLEMENTATION=1)
set(NYA_CUTLASS_TILE "64x64x32" CACHE STRING "CUTLASS threadblock tuning shape")
set_property(CACHE NYA_CUTLASS_TILE PROPERTY STRINGS 128x64x16 64x64x32 64x128x32 128x64x32)
if(NYA_CUTLASS_TILE STREQUAL "128x64x16")
    target_compile_definitions(fyodor-cutlass PRIVATE NYA_TILE_M=128 NYA_TILE_N=64 NYA_TILE_K=16)
elseif(NYA_CUTLASS_TILE STREQUAL "64x64x32")
    target_compile_definitions(fyodor-cutlass PRIVATE NYA_TILE_M=64 NYA_TILE_N=64 NYA_TILE_K=32)
elseif(NYA_CUTLASS_TILE STREQUAL "64x128x32")
    target_compile_definitions(fyodor-cutlass PRIVATE NYA_TILE_M=64 NYA_TILE_N=128 NYA_TILE_K=32)
elseif(NYA_CUTLASS_TILE STREQUAL "128x64x32")
    target_compile_definitions(fyodor-cutlass PRIVATE NYA_TILE_M=128 NYA_TILE_N=64 NYA_TILE_K=32)
else()
    message(FATAL_ERROR "Unknown NYA_CUTLASS_TILE")
endif()
target_include_directories(fyodor-cutlass PRIVATE "${NYA_CUTLASS_ROOT}/include")
foreach(license IN ITEMS LICENSE.txt LICENSE)
    if(EXISTS "${NYA_CUTLASS_ROOT}/${license}")
        configure_file("${NYA_CUTLASS_ROOT}/${license}" "${CMAKE_BINARY_DIR}/CUTLASS-LICENSE.txt" COPYONLY)
        break()
    endif()
endforeach()
set_target_properties(fyodor-cutlass PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Static POSITION_INDEPENDENT_CODE ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
# Embed Ampere code plus forward-compatible PTX unless the user selected targets.
# The bridge checks architecture eligibility before any kernel is dispatched.
if(NOT DEFINED NYA_CUTLASS_ARCHITECTURES)
    set(NYA_CUTLASS_ARCHITECTURES "80-real;80-virtual" CACHE STRING "CUTLASS CUDA architectures")
endif()
set_property(TARGET fyodor-cutlass PROPERTY CUDA_ARCHITECTURES "${NYA_CUTLASS_ARCHITECTURES}")
# NVCC forwards its optimization level to the host compiler. An unconditional
# -O3 therefore conflicts with MSVC Debug's /RTC1 runtime checks. Keep those
# checks usable while retaining optimized kernels in non-Debug configurations.
target_compile_options(fyodor-cutlass PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-O0>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:-O3>)
if(CMAKE_CUDA_SIMULATE_ID STREQUAL "MSVC")
    target_compile_options(fyodor-cutlass PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
    set_property(TARGET fyodor-cutlass PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()
message(STATUS "Optional CUTLASS bridge enabled; portable core remains C")
