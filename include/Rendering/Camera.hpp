#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"

namespace cb
{

class Camera
{
public:
	[[nodiscard]]
	__device__
	Ray get_ray(Float2 uv) const;

private:
	Float3 position;
	Float2 rotation;
	float forward_distance;
};

} // cb
