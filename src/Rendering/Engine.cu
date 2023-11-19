#include "Rendering/Engine.cuh"
#include "Rendering/Accumulator.hpp"
#include "Rendering/KernelLaunch.cuh"
#include "Utilities/CudaArray.hpp"

namespace cb
{

Engine::Engine() = default;

void Engine::change_resolution(const UInt2& new_resolution)
{
	if (resolution == new_resolution) return;
	resolution = new_resolution;

	uint32_t count = resolution.x() * resolution.y();
	accumulators = CudaArray<Accumulator>(count, true);
}

__global__
static void test_kernel(UInt2 resolution, CudaArray<Accumulator>::Accessor accumulators, float time)
{
	UInt2 position = get_thread_position2D();
	if (!(position < resolution)) return;
	Float2 uv(Float2(position) / Float2(resolution));

	float r = cosf(time + uv.x() + 0.0f);
	float g = cosf(time + uv.y() + 2.0f);
	float b = cosf(time + uv.x() + 4.0f);

	Float3 color = Float3(r, g, b) * 0.5f + Float3(0.5f);
	Accumulator& accumulator = accumulators[position.y() * resolution.x() + position.x()];
	accumulator = {};
	accumulator.insert(color);
}

__global__
static void output_kernel(UInt2 resolution, CudaArray<Accumulator>::Accessor accumulators, cudaSurfaceObject_t surface)
{
	UInt2 position = get_thread_position2D();
	if (!(position < resolution)) return;

	Float3 color = accumulators[position.y() * resolution.x() + position.x()].current();

	auto convert = [] __device__(float value) { return min((uint32_t)(sqrtf(value) * 256.0f), 255); };
	uint32_t value = 0xFF000000 | (convert(color.x()) << 16) | (convert(color.y()) << 8) | convert(color.z());
	surf2Dwrite(value, surface, static_cast<int>(position.x() * sizeof(uint32_t)), static_cast<int>(position.y()));
}

void Engine::output(cudaSurfaceObject_t surface_object)
{
	KernelLaunch launcher(resolution);
	launcher.launch(test_kernel, resolution, accumulators, time);
	launcher.launch(output_kernel, resolution, accumulators, surface_object);
}

} // cb
