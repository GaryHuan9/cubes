#include "Rendering/Engine.hpp"
#include "Rendering/Structures.hpp"
#include "Rendering/Kernels.hpp"
#include "Scenic/Camera.hpp"

namespace cb
{

class KernelLaunch
{
public:
	explicit KernelLaunch(unsigned int region, unsigned int block_size = 256)
	{
		block_count = dim3((region + block_size - 1) / block_size);
		this->block_size = dim3(block_size);
	}

	explicit KernelLaunch(const UInt2& region, unsigned int block_size = 16)
	{
		UInt2 count = (region + UInt2(block_size - 1)) / block_size;
		block_count = dim3(count.x(), count.y());
		this->block_size = dim3(block_size, block_size);
	}

	explicit KernelLaunch(const UInt3& region, unsigned int block_size = 8)
	{
		UInt3 count = (region + UInt3(block_size - 1)) / block_size;
		block_count = dim3(count.x(), count.y(), count.z());
		this->block_size = dim3(block_size, block_size, block_size);
	}

	template<typename Kernel, typename... Arguments>
	void launch(const Kernel& kernel, const Arguments& ... arguments)
	{
		kernel<<<block_count, block_size>>>(arguments...);
	}

private:
	dim3 block_count;
	dim3 block_size;
};

Engine::Engine()
{
	cuda_check(cudaMalloc(&camera, sizeof(Camera)));

	paths = CudaArray<Path>(Capacity);
	randoms = CudaArray<curandState>(Capacity);
	KernelLaunch(Capacity).launch(kernels::new_random, randoms);

	trace_queries = CudaVector<TraceQuery>(Capacity);
	material_queries = CudaVector<MaterialQuery>(Capacity);
	escape_packets = CudaVector<EscapedPacket>(Capacity);
}

Engine::~Engine() = default;

void Engine::change_resolution(const UInt2& new_resolution)
{
	if (resolution == new_resolution) return;
	resolution = new_resolution;

	uint32_t count = resolution.x() * resolution.y();
	accumulators = CudaArray<Accumulator>(count);
	accumulators.clear();
}

void Engine::change_camera(const Camera& new_camera)
{
	cuda_copy(camera, &new_camera);
	accumulators.clear();
}

void Engine::render()
{
	cuda_check(cudaDeviceSynchronize());

	trace_queries.clear();
	material_queries.clear();
	escape_packets.clear();

	KernelLaunch launcher(Capacity);

	index_start %= resolution.x() * resolution.y();
	launcher.launch(kernels::new_path, paths, resolution, index_start, camera, trace_queries);
	index_start += Capacity;

	for (size_t depth = 0; depth < 16; ++depth)
	{
		launcher.launch(kernels::trace, trace_queries);
		launcher.launch(kernels::shade, trace_queries, material_queries, escape_packets, randoms);

		cuda_check(cudaDeviceSynchronize());
		trace_queries.clear();

		launcher.launch(kernels::diffuse, material_queries);
		launcher.launch(kernels::advance, material_queries, trace_queries, paths, accumulators);
		launcher.launch(kernels::escaped, escape_packets, paths, accumulators);

		cuda_check(cudaDeviceSynchronize());
		material_queries.clear();
		escape_packets.clear();
	}
}

void Engine::output(cudaSurfaceObject_t surface_object) const
{
	KernelLaunch(resolution).launch(kernels::output, resolution, accumulators, surface_object);
}

}
