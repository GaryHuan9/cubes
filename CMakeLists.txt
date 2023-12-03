cmake_minimum_required(VERSION 3.20)
project(cubes LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cubes/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cubes/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/cubes/bin)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_compile_options(-lineinfo)
add_executable(cubes "")
add_subdirectory(ext)
add_subdirectory(src)

target_include_directories(cubes PRIVATE include)

add_custom_command(TARGET cubes POST_BUILD COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/ext/JetBrainsMono
        ${CMAKE_BINARY_DIR}/cubes/ext/JetBrainsMono
)

set_target_properties(cubes PROPERTIES CUDA_ARCHITECTURES native)
set_target_properties(cubes PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
