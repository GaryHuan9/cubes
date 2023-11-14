find_package(CUDAToolkit REQUIRED)
target_link_libraries(cubes CUDA::cudart)

target_sources(cubes PRIVATE
        main.cpp
        Application.cpp
        Component.cpp
        Renderer.cpp
)

add_subdirectory(kernels)
