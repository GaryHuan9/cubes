target_sources(cubes PRIVATE
        main.cpp
        Application.cpp
)

target_link_libraries(cubes ImGui-SFML::ImGui-SFML)
