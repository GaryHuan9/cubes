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

	HOST_DEVICE_NODISCARD
	Camera* get_camera()
	{
		return &camera;
	};

private:
	Camera camera;
};

}
