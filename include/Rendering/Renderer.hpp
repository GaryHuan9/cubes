#pragma once

#include "main.hpp"
#include "Component.hpp"
#include "Utilities/Vector.hpp"

class cudaGraphicsResource;

namespace cb
{

class Renderer : public Component
{
public:
	explicit Renderer(Application& application);

	[[nodiscard]]
	Engine* get_engine() const
	{
		return engine.get();
	}

	void initialize() override;
	void update(const Timer& timer) override;

private:
	void change_resolution(const UInt2& new_resolution);

	void create_resources();

	void clean_resources();

	UInt2 resolution;
	std::unique_ptr<Engine> engine;
	std::unique_ptr<sf::Sprite> sprite;
	std::unique_ptr<sf::Texture> texture;

	cudaGraphicsResource* graphics_resource{};
	cudaSurfaceObject_t surface_object{};
};

} // cb
