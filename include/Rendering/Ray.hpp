#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaUtilities.hpp"

namespace cb
{

class Ray
{
public:
	HOST_DEVICE
	Ray(const Float3& origin, const Float3& direction) : origin(origin), direction(direction) {}

	[[nodiscard]]
	HOST_DEVICE
	Float3 get_point(float length) const
	{
		return origin + direction * length;
	}

	const Float3 origin;
	const Float3 direction;
};

} // cb
