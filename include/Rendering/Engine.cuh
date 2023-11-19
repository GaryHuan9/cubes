#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaArray.hpp"

namespace cb
{

class Engine
{
public:
	Engine();

	void change_resolution(const UInt2& new_resolution);

	void output(cudaSurfaceObject_t surface_object);

	float time;

private:
	UInt2 resolution;
	CudaArray<Accumulator> accumulators;
};

} // cb
