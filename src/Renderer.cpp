#include "Renderer.hpp"
#include "Vector.hpp"

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace cb
{

void Renderer::initialize()
{

}

void Renderer::update()
{

}

void Renderer::recreate_resources(const Int2& resolution)
{
	//	if (graphics_resource != nullptr) cuda_check(cudaGraphicsUnregisterResource(graphics_resource));
	//
	//	texture->create(resolution.x, resolution.y);
	//	sprite->setTexture(*texture, true);
	//
	//	cuda_check(cudaGraphicsGLRegisterImage(
	//		&graphics_resource, texture->getNativeHandle(),
	//		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

} // cb
