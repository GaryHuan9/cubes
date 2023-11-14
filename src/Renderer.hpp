#pragma once

#include "main.hpp"
#include "Component.hpp"

class cudaGraphicsResource;

namespace cb
{

class Renderer : public Component
{
public:
	explicit Renderer(Application& application) : Component(application) {}

	void initialize() override;
	void update() override;

private:
	void recreate_resources(const Int2& resolution);

	cudaGraphicsResource* graphics_resource{};
};

} // cb
