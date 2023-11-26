#include "Rendering/Renderer.hpp"
#include "Rendering/Engine.hpp"
#include "Utilities/Vector.hpp"
#include "Application.hpp"

#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include "SFML/OpenGL.hpp"
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

namespace cb
{

Renderer::Renderer(Application& application) :
	Component(application),
	engine(std::make_unique<Engine>()),
	sprite(std::make_unique<sf::Sprite>()),
	texture(std::make_unique<sf::Texture>()) {}

void Renderer::initialize()
{
	sf::Vector2u size = window.getSize();
	UInt2 new_resolution(size.x, size.y);
	change_resolution(new_resolution);
}

void Renderer::update(const Timer& timer)
{
	sf::Vector2u size = window.getSize();
	UInt2 new_resolution(size.x, size.y);

	if (new_resolution != resolution) change_resolution(new_resolution);

	engine->render();
	engine->output(surface_object);
	cuda_check(cudaDeviceSynchronize());
	window.draw(*sprite);
}

void Renderer::change_resolution(const UInt2& new_resolution)
{
	resolution = new_resolution;
	engine->change_resolution(resolution);
	clean_resources();

	//Create CUDA resources for interop
	texture->create(resolution.x(), resolution.y());
	sprite->setTexture(*texture, true);
	create_resources();
}

void Renderer::create_resources()
{
	cuda_check(cudaGraphicsGLRegisterImage(
		&graphics_resource, texture->getNativeHandle(),
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	cudaResourceDesc description{ cudaResourceTypeArray };
	cudaArray_t* array = &description.res.array.array;
	cuda_check(cudaGraphicsMapResources(1, &graphics_resource));
	cuda_check(cudaGraphicsSubResourceGetMappedArray(array, graphics_resource, 0, 0));

	cuda_check(cudaCreateSurfaceObject(&surface_object, &description));
}

void Renderer::clean_resources()
{
	if (graphics_resource == nullptr) return;

	cuda_check(cudaDestroySurfaceObject(surface_object));
	cuda_check(cudaGraphicsUnmapResources(1, &graphics_resource));
	cuda_check(cudaGraphicsUnregisterResource(graphics_resource));

	graphics_resource = nullptr;
	surface_object = {};
}

} // cb
