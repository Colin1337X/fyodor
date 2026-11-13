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
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gemm.cu")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/gemm.cu" nya_gemm_hex HEX)
string(APPEND nya_cuda_hex "0a" "${nya_gemm_hex}")
set(nya_training_common_hex "${nya_cuda_hex}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/resident.cu")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/resident.cu" nya_resident_hex HEX)
string(APPEND nya_cuda_hex "0a" "${nya_resident_hex}")
option(NYA_ENABLE_OPTCPP "Enable physically removable root optcpp components" ON)
option(NYA_ENABLE_PORTABLE_ATTENTION "Compile experimental portable attention kernels" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_PORTABLE_ATTENTION AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/portable_f32.cu")
    set(nya_portable_source "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/portable_f32.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_portable_source}")
    file(READ "${nya_portable_source}" nya_portable_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_portable_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_PORTABLE_ATTENTION=1)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp.cu")
        set(nya_fdpp_source "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp.cu")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_fdpp_source}")
        file(READ "${nya_fdpp_source}" nya_fdpp_hex HEX)
        string(APPEND nya_cuda_hex "0a" "${nya_fdpp_hex}")
        target_compile_definitions(nya-engine PRIVATE NYA_FLASH_DECODING_PP=1)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_warp.cu")
            set(nya_fdpp_warp_source "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_warp.cu")
            set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_fdpp_warp_source}")
            file(READ "${nya_fdpp_warp_source}" nya_fdpp_warp_hex HEX)
            string(APPEND nya_cuda_hex "0a" "${nya_fdpp_warp_hex}")
            target_compile_definitions(nya-engine PRIVATE NYA_FLASH_DECODING_PP_WARP=1)
            if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_specialized.cu")
                set(nya_fdpp_specialized "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_specialized.cu")
                set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_fdpp_specialized}")
                file(READ "${nya_fdpp_specialized}" nya_fdpp_specialized_hex HEX)
                string(APPEND nya_cuda_hex "0a" "${nya_fdpp_specialized_hex}")
                target_compile_definitions(nya-engine PRIVATE NYA_FLASH_DECODING_PP_SPECIALIZED=1)
            endif()
        endif()
    endif()

elseif(NYA_ENABLE_PORTABLE_ATTENTION)
    message(STATUS "Optional attention sources disabled or absent: retaining native attention")
endif()

option(NYA_ENABLE_FIXED_GATE "Compile experimental fixed-width decode gate" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_FIXED_GATE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/fixed_width_gate.cu")
    set(nya_fixed_gate "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/fixed_width_gate.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_fixed_gate}")
    file(READ "${nya_fixed_gate}" nya_fixed_gate_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_fixed_gate_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_FIXED_GATE=1)
endif()

option(NYA_ENABLE_INTEGER_GATE "Compile experimental integer scale-product decode gate" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_INTEGER_GATE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/integer_scale_gate.cu")
    set(nya_integer_gate "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/integer_scale_gate.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_integer_gate}")
    file(READ "${nya_integer_gate}" nya_integer_gate_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_integer_gate_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_INTEGER_GATE=1)
endif()

option(NYA_ENABLE_INTEGER_DOT "Compile experimental Q4 integer scale-product decode dots" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_INTEGER_DOT AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/integer_scale_dot.cu")
    set(nya_integer_dot "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/integer_scale_dot.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_integer_dot}")
    file(READ "${nya_integer_dot}" nya_integer_dot_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_integer_dot_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_INTEGER_DOT=1)
endif()

option(NYA_ENABLE_INTEGER_Q6 "Compile experimental signed Q6 integer-product decode dots" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_INTEGER_Q6 AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/integer_scale_q6.cu")
    set(nya_integer_q6 "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/integer_scale_q6.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_integer_q6}")
    file(READ "${nya_integer_q6}" nya_integer_q6_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_integer_q6_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_INTEGER_Q6=1)
endif()

option(NYA_ENABLE_BYTE_GATE "Compile experimental aligned byte-metadata Q4 gate" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_BYTE_GATE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/byte_scale_gate_aligned.cu")
    set(nya_byte_gate "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/byte_scale_gate_aligned.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_byte_gate}")
    file(READ "${nya_byte_gate}" nya_byte_gate_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_byte_gate_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_BYTE_GATE=1)
endif()

option(NYA_ENABLE_BYTE_DOT "Compile experimental aligned byte-metadata Q4 projections" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_BYTE_DOT AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/byte_scale_dot.cu")
    set(nya_byte_dot "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/byte_scale_dot.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_byte_dot}")
    file(READ "${nya_byte_dot}" nya_byte_dot_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_byte_dot_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_BYTE_DOT=1)
endif()

option(NYA_ENABLE_GEMMA_GATE "Compile experimental Gemma packed Q4 GELU gate" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_GEMMA_GATE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/gemma_gate.cu")
    set(nya_gemma_gate "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/gemma_gate.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_gemma_gate}")
    file(READ "${nya_gemma_gate}" nya_gemma_gate_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_gemma_gate_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_GEMMA_GATE=1)
endif()

option(NYA_ENABLE_WIDE_VALUE "Compile optional wide attention value reduction" OFF)
option(NYA_ENABLE_WIDE_BATCH_SPLIT "Compile optional batched split wide attention" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_WIDE_VALUE AND NYA_ENABLE_WIDE_BATCH_SPLIT AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_value16.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_batch_split16.cu")
    set(nya_wide_batch_split "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_batch_split16.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_wide_batch_split}")
    file(READ "${nya_wide_batch_split}" nya_wide_batch_split_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_wide_batch_split_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_WIDE_BATCH_SPLIT=1)
endif()

if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_WIDE_VALUE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_value16.cu")
    set(nya_wide_value "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_value16.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_wide_value}")
    file(READ "${nya_wide_value}" nya_wide_value_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_wide_value_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_WIDE_VALUE=1)
endif()

option(NYA_ENABLE_SMALL_BATCH "Compile optional narrow-batch row-dot scheduling" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_SMALL_BATCH AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/small_batch.cu")
    set(nya_small_batch "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/small_batch.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_small_batch}")
    file(READ "${nya_small_batch}" nya_small_batch_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_small_batch_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_SMALL_BATCH=1)
endif()

option(NYA_ENABLE_SMALL_BATCH_REUSE "Compile optional narrow-batch weight reuse" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_SMALL_BATCH_REUSE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/small_batch_reuse.cu")
    set(nya_small_reuse "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/small_batch_reuse.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_small_reuse}")
    file(READ "${nya_small_reuse}" nya_small_reuse_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_small_reuse_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_SMALL_BATCH_REUSE=1)
endif()

option(NYA_ENABLE_ALIGNED_PREFIX "Compile experimental serial-aligned Gemma prefix execution" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_ALIGNED_PREFIX AND NYA_ENABLE_PORTABLE_ATTENTION AND
   NYA_ENABLE_WIDE_VALUE AND NYA_ENABLE_WIDE_BATCH_SPLIT AND
   NYA_ENABLE_SMALL_BATCH AND NYA_ENABLE_SMALL_BATCH_REUSE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/portable_f32.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_warp.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_batch.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_value16.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/wide_batch_split16.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/small_batch.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/small_batch_reuse.cu")
    set(nya_fdpp_batch "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/attention/flash_decoding_pp_batch.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_fdpp_batch}")
    file(READ "${nya_fdpp_batch}" nya_fdpp_batch_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_fdpp_batch_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_ALIGNED_PREFIX=1)
elseif(NYA_ENABLE_ALIGNED_PREFIX)
    message(STATUS "Aligned prefix dependencies disabled or absent: retaining existing prefix execution")
endif()

option(NYA_ENABLE_INTERLEAVED_TWO "Compile optional interleaved batch-two Q6 weight reuse" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_INTERLEAVED_TWO AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_two.cu")
    set(nya_interleaved_two "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_two.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_interleaved_two}")
    file(READ "${nya_interleaved_two}" nya_interleaved_two_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_interleaved_two_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_INTERLEAVED_TWO=1)
endif()

option(NYA_ENABLE_INTERLEAVED_REUSE "Compile optional interleaved batch-four weight reuse" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_INTERLEAVED_REUSE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_reuse.cu")
    set(nya_interleaved_reuse "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_reuse.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_interleaved_reuse}")
    file(READ "${nya_interleaved_reuse}" nya_interleaved_reuse_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_interleaved_reuse_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_INTERLEAVED_REUSE=1)
endif()

option(NYA_ENABLE_INTERLEAVED_INTEGER "Compile experimental Q4 integer batch-four coefficients" OFF)
if(NYA_ENABLE_OPTCPP AND NYA_ENABLE_INTERLEAVED_INTEGER AND NYA_ENABLE_INTERLEAVED_REUSE AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_reuse.cu" AND
   EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_integer.cu")
    set(nya_interleaved_integer "${CMAKE_CURRENT_SOURCE_DIR}/../optcpp/matvec/interleaved_integer.cu")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${nya_interleaved_integer}")
    file(READ "${nya_interleaved_integer}" nya_interleaved_integer_hex HEX)
    string(APPEND nya_cuda_hex "0a" "${nya_interleaved_integer_hex}")
    target_compile_definitions(nya-engine PRIVATE NYA_INTERLEAVED_INTEGER=1)
endif()

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/training.cu")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/training.cu" nya_training_hex HEX)
string(PREPEND nya_training_hex "${nya_training_common_hex}" "0a")

string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," nya_cuda_bytes "${nya_cuda_hex}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_source.h" "static const unsigned char nya_cuda_source[] = {${nya_cuda_bytes}0};\n")
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," nya_training_bytes "${nya_training_hex}")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/cuda_source.h" "static const unsigned char nya_cuda_training_source[] = {${nya_training_bytes}0};\n")
file(TO_CMAKE_PATH "${NYA_CUDA_NVRTC_LIBRARY}" nya_nvrtc_default)
string(REPLACE "\"" "\\\"" nya_nvrtc_default "${nya_nvrtc_default}")
file(APPEND "${CMAKE_CURRENT_BINARY_DIR}/cuda_source.h" "static const char nya_nvrtc_default[] = \"${nya_nvrtc_default}\";\n")
target_sources(nya-engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/cuda.c")
target_include_directories(nya-engine PRIVATE "${NYA_CUDA_INCLUDE_DIR}" "${NYA_NVRTC_INCLUDE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(nya-engine PRIVATE NYA_ENABLE_CUDA=1)
set(NYA_CUDA_AVAILABLE TRUE PARENT_SCOPE)
message(STATUS "CUDA driver/NVRTC provider enabled (C host compiler; runtime CUDA compilation)")

option(NYA_ENABLE_CUTLASS "Enable the separately removable CUTLASS adapter" ON)
if(NYA_ENABLE_CUTLASS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cutlass/CMakeLists.txt")
    add_subdirectory(cutlass)
elseif(NYA_ENABLE_CUTLASS)
    message(STATUS "CUDA/cutlass directory absent: retaining native CUDA kernels")
endif()
