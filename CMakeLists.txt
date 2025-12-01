cmake_minimum_required(VERSION 3.20)
project(imemdb VERSION 0.1.0 LANGUAGES CXX CUDA)

# 1. GPU Architecture setup 
# locally i have only rtx 3060 which is compute compatible with 8.6 (Ampere)
# using native tries to detect it, meanwhile '86' will force it
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#2. Standard & Flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#cmpl and optm
add_compile_options(-Wall -Wextra)

# Debug Flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")

#Release flags (For max performance)
#-march=native: optimizes for ryzen 5900x
# -O3 Max optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fvisibility=hidden -flto=auto -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG --use_fast_math")

#3. Core Library (Engine)
add_library(inmemdb SHARED
    # Memory & Layout
    src/storage/layout.cu
    
    # GPU Kernels
    src/gpu/hash_kernel.cu
    src/gpu/insert_kernel.cu
    src/gpu/lookup_kernel.cu
    
    # CPU Router / Execution
    src/execution/hash_router.cpp
)

target_include_directories(inmemdb PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR} 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(inmemdb PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    CUDA_SEPARABLE_COMPILATION ON
)

if(UNIX AND NOT APPLE)
    target_compile_options(inmemdb PRIVATE -fvisibility=hidden)
endif()

#4. Main executable 
add_executable(gpu_db_engine src/main.cpp)

#5. Tests & Benchmarks (Future) to be added

#6. Installation
install(TARGETS inmemdb
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

