cmake_minimum_required(VERSION 3.25)
project(fyodor VERSION 0.3.0 LANGUAGES C)

# Both entry points compile only C; the backend can also build by itself.
include(CTest)
add_subdirectory(backend)
