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

	void start_render(uint32_t start_index);

	void start_render()
	{
		for (size_t i = 0; i < 10; ++i)
		{
			current_index %= resolution.product();
			start_render(current_index);
			current_index += Capacity;
		}
	}

	void render(uint32_t samples_per_pixel)
	{
		uint64_t iteration = static_cast<uint64_t>(samples_per_pixel) * resolution.product() / Capacity;
		uint32_t start_index = 0;

		for (uint64_t i = 0; i < iteration; ++i)
		{
			start_render(start_index);
			start_index += Capacity;
			start_index %= resolution.product();
		}

		cuda_check(cudaDeviceSynchronize());
	}

	void output(cudaSurfaceObject_t surface_object) const;

	void output(const std::string& filename) const;

private:
	static constexpr size_t Capacity = 1024 * 1024;

	UInt2 resolution;
	Camera* camera{};

	uint32_t current_index{};

	CudaArray<Path> paths;
	CudaArray<Accumulator> accumulators;
	CudaArray<curandState> randoms;

	CudaVector<TraceQuery> trace_queries;
	CudaVector<MaterialQuery> material_queries;
	CudaVector<EscapedPacket> escape_packets;
};

}
