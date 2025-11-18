cmake_minimum_required(VERSION 3.20)
project(inmemdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

add_library(inmemdb
    src/storage/hash_table_backend.cpp
)

target_include_directories(inmemdb PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(inmemdb_test
    src/main.cpp
)

target_link_libraries(inmemdb_test PRIVATE inmemdb)