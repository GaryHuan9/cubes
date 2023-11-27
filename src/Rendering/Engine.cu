#include "Rendering/Engine.hpp"
#include "Rendering/Packets.hpp"
#include "Rendering/Kernels.hpp"
#include "Rendering/Accumulator.hpp"
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
	new_path_packets = CudaArray<NewPathPackets>(Capacity);
	trace_packets = CudaVector<TracePackets>(Capacity);
	hit_packets = CudaArray<HitPacket>(Capacity);
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
	trace_packets.clear();

	scan_offset %= resolution.x() * resolution.y();
	KernelLaunch(Capacity).launch(kernels::render_begin, resolution, scan_offset, Capacity, new_path_packets);
	scan_offset += Capacity;

	KernelLaunch(Capacity).launch(kernels::new_path, resolution, camera, new_path_packets, trace_packets);
	KernelLaunch(Capacity).launch(kernels::trace_rays, trace_packets, hit_packets);
	KernelLaunch(Capacity).launch(kernels::render_end, resolution, trace_packets, hit_packets, accumulators);
}

void Engine::output(cudaSurfaceObject_t surface_object) const
{
	//	KernelLaunch(resolution).launch(kernels::test, resolution, accumulators, 0.0f);
	KernelLaunch(resolution).launch(kernels::output, resolution, accumulators, surface_object);
}

} // cb
