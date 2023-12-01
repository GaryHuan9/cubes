#include "Rendering/Engine.hpp"
#include "Rendering/Structures.hpp"
#include "Rendering/Kernels.hpp"
#include "Rendering/Launcher.hpp"
#include "Scenic/Camera.hpp"
#include "Utilities/Image.hpp"

namespace cb
{

Engine::Engine()
{
	cuda_check(cudaMalloc(&camera, sizeof(Camera)));

	paths = CudaArray<Path>(Capacity);
	randoms = CudaArray<curandState>(Capacity);
	Launcher(kernels::new_random, Capacity).launch(randoms);

	trace_queries = CudaVector<TraceQuery>(Capacity);
	material_queries = CudaVector<MaterialQuery>(Capacity);
	escape_packets = CudaVector<EscapedPacket>(Capacity);
}

Engine::~Engine() = default;

void Engine::change_resolution(const UInt2& new_resolution)
{
	if (resolution == new_resolution) return;
	resolution = new_resolution;

	uint32_t count = resolution.product();
	accumulators = CudaArray<Accumulator>(count);
	reset_render();
}

void Engine::change_camera(const Camera& new_camera)
{
	cuda_copy(camera, &new_camera);
	reset_render();
}

void Engine::reset_render()
{
	accumulators.clear();
	current_index = resolution.product() / 2;
	current_index -= Capacity / 2;
}

void Engine::start_render(uint32_t start_index)
{
	auto clear_list = []<typename... Ts>(CudaVector<Ts>& ... lists) { Launcher(kernels::list_clear<Ts...>).launch(lists...); };

	clear_list(trace_queries, material_queries, escape_packets);
	Launcher(kernels::new_path, Capacity).launch(paths, resolution, start_index, randoms, camera, trace_queries);

	for (size_t depth = 0; depth < 16; ++depth)
	{
		Launcher(kernels::trace, Capacity).launch(trace_queries, material_queries, escape_packets);
		Launcher(kernels::pre_material, Capacity).launch(material_queries, randoms);
		clear_list(trace_queries);

		Launcher(kernels::diffuse, Capacity).launch(material_queries, paths, trace_queries, DiffuseParameters(Float3(0.8f)));
		//		Launcher(kernels::conductor, Capacity).launch(material_queries, paths, trace_queries, ConductorParameters(Float3(0.8f), 0.1f));
		Launcher(kernels::escaped, Capacity).launch(escape_packets, paths);
		clear_list(material_queries, escape_packets);
	}

	Launcher(kernels::accumulate, Capacity).launch(paths, start_index, accumulators);
}

void Engine::output(cudaSurfaceObject_t surface_object) const
{
	Launcher(kernels::output_surface, resolution.product()).launch(resolution, accumulators, surface_object);
}

void Engine::output(const std::string& filename) const
{
	size_t count = resolution.product();
	CudaArray<uint32_t> buffer_device(count);
	auto buffer_host = std::make_unique<uint32_t[]>(count);

	Launcher(kernels::output_buffer, resolution.product()).launch(resolution, accumulators, buffer_device);
	cuda_check(cudaMemcpy(buffer_host.get(), buffer_device.data(), count * sizeof(uint32_t), cudaMemcpyDefault));
	Image::write_png(filename, resolution.x(), resolution.y(), buffer_host.get());
}

}
