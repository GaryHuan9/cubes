#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaArray.hpp"
#include "Utilities/CudaVector.hpp"

#include <curand_kernel.h>

namespace cb
{

class Engine
{
public:
	Engine();
	~Engine();

	void change_resolution(const UInt2& new_resolution);

	void change_camera(const Camera& new_camera);

	void reset_render();

	void render();

	void output(cudaSurfaceObject_t surface_object) const;

private:
	static constexpr size_t Capacity = 1024 * 1024;

	UInt2 resolution;
	Camera* camera{};

	uint32_t index_start{};

	CudaArray<Path> paths;
	CudaArray<Accumulator> accumulators;
	CudaArray<curandState> randoms;

	CudaVector<TraceQuery> trace_queries;
	CudaVector<MaterialQuery> material_queries;
	CudaVector<EscapedPacket> escape_packets;
};

}
