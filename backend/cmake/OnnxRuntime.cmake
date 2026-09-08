# This optional adapter compiles against the C ABI. ONNX Runtime itself is an
# external implementation; omit this option for a self-contained C backend.
set(NYA_ONNXRUNTIME_ROOT "" CACHE PATH "Extracted ONNX Runtime SDK/NuGet package")
set(NYA_ONNXRUNTIME_VERSION "1.26.0")
if(NOT NYA_ONNXRUNTIME_ROOT)
    if(NOT NYA_FETCH_ONNXRUNTIME)
        message(FATAL_ERROR "Set NYA_ONNXRUNTIME_ROOT or explicitly enable NYA_FETCH_ONNXRUNTIME")
    endif()
    include(FetchContent)
    FetchContent_Declare(nya_onnxruntime_sdk
        URL "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime/${NYA_ONNXRUNTIME_VERSION}/microsoft.ml.onnxruntime.${NYA_ONNXRUNTIME_VERSION}.nupkg"
        URL_HASH SHA256=50cc3772668f04b8373ad65a36793f94699bc4e818f6e691fc68f1578c38ce42
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(nya_onnxruntime_sdk)
    set(NYA_ONNXRUNTIME_ROOT "${nya_onnxruntime_sdk_SOURCE_DIR}")
endif()

# Pick only the target architecture, never the first binary from a mixed SDK.
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" nya_arch)
if(CMAKE_GENERATOR_PLATFORM)
    string(TOLOWER "${CMAKE_GENERATOR_PLATFORM}" nya_arch)
endif()
if(nya_arch MATCHES "^(amd64|x86_64|x64)$")
    set(nya_arch x64)
elseif(nya_arch MATCHES "^(aarch64|arm64)$")
    set(nya_arch arm64)
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4 AND WIN32)
    set(nya_arch x86)
else()
    message(FATAL_ERROR "Unsupported ONNX Runtime package architecture: ${nya_arch}")
endif()
if(WIN32)
    set(nya_ort_platform win)
    set(nya_ort_library onnxruntime.dll)
elseif(APPLE)
    set(nya_ort_platform osx)
    set(nya_ort_library libonnxruntime.dylib)
else()
    set(nya_ort_platform linux)
    set(nya_ort_library libonnxruntime.so)
endif()
# Noncached discovery honors SDK changes on subsequent configure operations.
find_path(NYA_ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_c_api.h
    PATHS "${NYA_ONNXRUNTIME_ROOT}/build/native/include" "${NYA_ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)
find_file(NYA_ONNXRUNTIME_SHARED_LIBRARY NAMES "${nya_ort_library}"
    PATHS "${NYA_ONNXRUNTIME_ROOT}/runtimes/${nya_ort_platform}-${nya_arch}/native"
        "${NYA_ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)
target_include_directories(nya-engine SYSTEM PRIVATE "${NYA_ONNXRUNTIME_INCLUDE_DIR}")
target_compile_definitions(nya-engine PRIVATE NYA_ENABLE_ONNXRUNTIME=1)
