cmake_minimum_required(VERSION 3.20)
project(inmemdb VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Werror)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -fvisibility=hidden -flto=auto -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -arch=sm_80 -DNDEBUG")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(inmemdb SHARED
    src/storage/hash_table_backend.cpp
    src/gpu/hash_kernel.cu
    src/storage/gpu_memory.cu
    src/storage/layout.cu
)

target_include_directories(inmemdb PUBLIC 
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

add_executable(inmemdb_test
    src/main.cpp
)
target_link_libraries(inmemdb_test PRIVATE inmemdb)

add_executable(test_hash_table
    tests/test_hash_table.cpp
)
target_link_libraries(test_hash_table PRIVATE inmemdb)

add_executable(bench_compare
    benchmarks/bench_compare.cpp
)
target_link_libraries(bench_compare PRIVATE inmemdb)

enable_testing()
add_test(NAME hash_table_tests COMMAND test_hash_table)
add_test(NAME benchmark COMMAND bench_compare)

install(TARGETS inmemdb
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/inmemdb.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/inmemdb.pc
    @ONLY
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/inmemdb.pc
    DESTINATION lib/pkgconfig
)
