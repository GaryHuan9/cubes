#pragma once

#include "main.hpp"
#include "Ray.hpp"

namespace cb
{

class NewPathPackets
{
public:
	const Int2 pixel;
};

class TracePackets
{
public:
	__device__
	TracePackets(const Int2& pixel, Ray ray) : pixel(pixel), ray(cuda_move(ray)) {}

	const Int2 pixel;
	const Ray ray;
};

} // cb
