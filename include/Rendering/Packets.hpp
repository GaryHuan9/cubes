#pragma once

#include "main.hpp"
#include "Ray.hpp"

namespace cb
{

class NewPathPackets
{
public:
	__device__
	explicit NewPathPackets(const UInt2& pixel) : pixel(pixel) {}

	const UInt2 pixel;
};

class TracePackets
{
public:
	__device__
	TracePackets(const UInt2& pixel, Ray ray) : pixel(pixel), ray(cuda_move(ray)) {}

	const UInt2 pixel;
	const Ray ray;
};

class HitPacket
{
public:
	__device__
	HitPacket() : distance(INFINITY) {}

	__device__
	HitPacket(float distance, const Float3& normal) : distance(distance), normal(normal) {}

	[[nodiscard]]
	HOST_DEVICE
	bool hit() const
	{
		return isfinite(distance);
	}

	const float distance;
	const Float3 normal;
};

} // cb
