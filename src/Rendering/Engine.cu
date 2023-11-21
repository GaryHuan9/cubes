#include "Rendering/Engine.hpp"
#include "Rendering/Packets.hpp"
#include "Rendering/Accumulator.hpp"
#include "Scenic/Camera.hpp"
#include "Scenic/Scene.hpp"
#include "Rendering/Renderer.hpp"

template<typename T>
using Array = cb::CudaArray<T>::Accessor;

template<typename T>
using List = cb::CudaVector<T>::Accessor;

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

Engine::Engine() = default;
Engine::~Engine() = default;

void Engine::change_resolution(const UInt2& new_resolution)
{
	std::vector<int> a;

	if (resolution == new_resolution) return;
	resolution = new_resolution;

	uint32_t count = resolution.x() * resolution.y();
	accumulators = CudaArray<Accumulator>(count, true);
}

namespace kernels
{

__global__
static void test(UInt2 resolution, Array <Accumulator> accumulators, float time);

__global__
static void new_path(UInt2 resolution, List <NewPathPackets> packets, List <TracePackets> results);

__global__
static void output(UInt2 resolution, Array <Accumulator> accumulators, cudaSurfaceObject_t surface);

} // kernels

void Engine::output(cudaSurfaceObject_t surface_object)
{
	KernelLaunch launcher(resolution);
	launcher.launch(kernels::test, resolution, accumulators, 0.0f);
	launcher.launch(kernels::output, resolution, accumulators, surface_object);
}

} // cb

namespace cb::kernels
{

__global__
static void test(UInt2 resolution, Array <Accumulator> accumulators, float time)
{
	UInt2 index = get_thread_index2D();
	if (!(index < resolution)) return;
	Float2 uv(Float2(index) / Float2(resolution));

	float r = cosf(time + uv.x() + 0.0f);
	float g = cosf(time + uv.y() + 2.0f);
	float b = cosf(time + uv.x() + 4.0f);

	Float3 color = Float3(r, g, b) * 0.5f + Float3(0.5f);
	Accumulator& accumulator = accumulators[index.y() * resolution.x() + index.x()];
	accumulator = {};
	accumulator.insert(color);
}

__global__
static void new_path(UInt2 resolution, const List <NewPathPackets> packets, List <TracePackets> results)
{
	uint32_t index = get_thread_index1D();
	if (index < packets.size()) return;
	const auto& packet = packets[index];

	Camera* camera = nullptr;

	Float2 uv = Float2(packet.pixel) - Float2(resolution) * 0.5f;
	float multiplier = 1.0f / static_cast<float>(resolution.x());
	results.emplace_back(packet.pixel, camera->get_ray(uv * multiplier));
}

__global__
static void output(UInt2 resolution, Array <Accumulator> accumulators, cudaSurfaceObject_t surface)
{
	UInt2 position = get_thread_index2D();
	if (!(position < resolution)) return;

	Float3 color = accumulators[position.y() * resolution.x() + position.x()].current();

	auto convert = [] __device__(float value) { return min((uint32_t)(sqrtf(value) * 256.0f), 255); };
	uint32_t value = 0xFF000000 | (convert(color.x()) << 16) | (convert(color.y()) << 8) | convert(color.z());
	surf2Dwrite(value, surface, static_cast<int>(position.x() * sizeof(uint32_t)), static_cast<int>(position.y()));
}

} // cb::kernel
