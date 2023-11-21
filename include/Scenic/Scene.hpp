#pragma once

#include "main.hpp"

#include <cuda_runtime.h>

namespace cb
{

class Scene
{
public:
	Scene();

	[[nodiscard]]
	Camera* get_camera() const
	{
		return camera.get();
	};

private:
	std::unique_ptr<Camera> camera;
};

} // cb
