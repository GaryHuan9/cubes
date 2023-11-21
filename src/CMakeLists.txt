find_package(CUDAToolkit REQUIRED)
target_link_libraries(cubes CUDA::cudart)

target_sources(cubes PRIVATE
        main.cpp
        Application.cpp
        Component.cpp
)

add_subdirectory(Rendering)
add_subdirectory(Utilities)
add_subdirectory(Scenic)
