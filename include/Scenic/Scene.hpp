#pragma once

#include "main.hpp"
#include "Camera.hpp"

#include <cuda_runtime.h>

namespace cb
{

class Scene
{
public:
	Scene();

	[[nodiscard]]
	HOST_DEVICE
	Camera* get_camera()
	{
		return &camera;
	};

private:
	Camera camera;
};

} // cb
