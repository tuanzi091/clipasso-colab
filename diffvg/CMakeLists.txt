cmake_minimum_required(VERSION 3.12)

project(diffvg VERSION 0.0.1 DESCRIPTION "Differentiable Vector Graphics")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)

# pybind11: prefer an explicit -Dpybind11_DIR, then the pip package, then the bundled source.
if(NOT pybind11_DIR)
    execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                    OUTPUT_VARIABLE pybind11_DIR
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET
                    RESULT_VARIABLE PYBIND11_CMAKEDIR_RESULT)
endif()
if(pybind11_DIR AND EXISTS "${pybind11_DIR}")
    find_package(pybind11 CONFIG REQUIRED)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/CMakeLists.txt")
    message(STATUS "pybind11 cmake dir not found, building bundled pybind11")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
else()
    message(FATAL_ERROR "pybind11 not found. Run: pip install pybind11==2.13.6")
endif()

option(DIFFVG_CUDA "Build diffvg with GPU code path?" OFF)

if(DIFFVG_CUDA)
    message(STATUS "Build with CUDA support")
    find_package(CUDA 10 REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT WIN32)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++17")
    endif()
else()
    message(STATUS "Build without CUDA support")
endif()

include_directories(${Python_INCLUDE_DIRS})
if(DIFFVG_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    # CUDA 13 moves CCCL headers to include/cccl; keep them visible to the host compiler.
    foreach(cuda_inc ${CUDA_INCLUDE_DIRS})
        if(EXISTS "${cuda_inc}/cccl")
            include_directories("${cuda_inc}/cccl")
        endif()
    endforeach()
endif()

if(NOT MSVC)
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
  # Force include limits and cstdlib headers to ensure all standard macros are defined
  add_compile_options(/FI"limits")
  add_compile_options(/FI"cstdlib")
  # Also define macros directly as a fallback
  add_compile_definitions(MB_LEN_MAX=16 INT_MAX=2147483647 INT_MIN=-2147483648 CHAR_BIT=8 UINT_MAX=4294967295 ULONG_MAX=4294967295 ULLONG_MAX=18446744073709551615 LLONG_MIN=-9223372036854775808LL LLONG_MAX=9223372036854775807LL USHRT_MAX=65535 SHRT_MAX=32767 SHRT_MIN=-32768 LONG_MAX=2147483647L LONG_MIN=-2147483648L SCHAR_MAX=127 SCHAR_MIN=-128 UCHAR_MAX=255 CHAR_MIN=-128 CHAR_MAX=127)
endif()

set(SRCS atomic.h
         color.h
         cdf.h
         cuda_utils.h
         diffvg.h
         edge_query.h
         filter.h
         matrix.h
         parallel.h
         pcg.h
         ptr.h
         sample_boundary.h
         scene.h
         shape.h
         solve.h
         vector.h
         within_distance.h
         winding_number.h
         atomic.cpp
         color.cpp
         diffvg.cpp
         parallel.cpp
         scene.cpp
         shape.cpp)

if(DIFFVG_CUDA)
    add_compile_definitions(COMPILE_WITH_CUDA)
    set_source_files_properties(
        diffvg.cpp
        scene.cpp
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    cuda_add_library(diffvg MODULE ${SRCS})
else()
    add_library(diffvg MODULE ${SRCS})
endif()

if(DIFFVG_CUDA)
    # cuda_add_library internally links CUDA libraries with the plain signature,
    # so this target must also use the plain signature.
    target_link_libraries(diffvg pybind11::module)
else()
    target_link_libraries(diffvg PRIVATE pybind11::module)
endif()

if(APPLE)
    set_target_properties(diffvg PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
endif()

set_target_properties(diffvg PROPERTIES PREFIX "" OUTPUT_NAME "diffvg")

if(NOT PYTHON_SITE_PACKAGES)
  set(PYTHON_SITE_PACKAGES "${CMAKE_CURRENT_BINARY_DIR}/pydiffvg")
endif()

install(TARGETS diffvg
  LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}/pydiffvg"
)