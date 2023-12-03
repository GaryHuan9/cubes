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
	cuda_malloc(camera);

	paths = CudaArray<Path>(Capacity);
	randoms = CudaArray<curandState>(Capacity);
	Launcher(kernels::new_random, Capacity).launch(randoms);

	trace_queries = CudaVector<TraceQuery>(Capacity);
	material_queries = CudaVector<MaterialQuery>(Capacity);
	escape_packets = CudaVector<EscapedPacket>(Capacity);

	materials.emplace_back(DiffuseParameters(Float3(0.8f)));
	materials.emplace_back(ConductorParameters(Float3(0.8f), 0.1f));
	materials.emplace_back(DielectricParameters(Float3(0.8f), 1.5f));
	materials.emplace_back(EmissiveParameters(Float3(2.0f)));

	for (size_t i = 0; i < materials.size(); ++i) material_indices.emplace_back(Capacity);
	material_indices_device = CudaArray<CudaVector<uint32_t>::Accessor>(materials.size());

	//TODO: this copying is VERY unsafe! But doing this otherwise is super messy
	void* destination = reinterpret_cast<void*>(material_indices_device.data());
	size_t length = sizeof(decltype(material_indices)::value_type) * material_indices.size();
	cuda_check(cudaMemcpy(destination, material_indices.data(), length, cudaMemcpyDefault));
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
	trace_queries.clear_async();
	material_queries.clear_async();
	escape_packets.clear_async();
	Launcher(kernels::new_path, Capacity).launch(paths, resolution, start_index, randoms, camera, trace_queries);

	for (size_t depth = 0; depth < 16; ++depth)
	{
		Launcher(kernels::trace, Capacity).launch(trace_queries, material_queries, escape_packets);
		trace_queries.clear_async();

		Launcher(kernels::pre_material, Capacity).launch(material_queries, material_indices_device, randoms);

		for (size_t i = 0; i < materials.size(); ++i)
		{
			const auto& material = materials[i];
			auto& indices = material_indices[i];

			auto try_launch = [&]<typename Kernel, typename Parameters>(Kernel kernel)
			{
				if (!std::holds_alternative<Parameters>(material)) return;
				Launcher(kernel, Capacity).launch(indices, material_queries, paths, trace_queries, std::get<Parameters>(material));
			};

			try_launch.template operator()<decltype(kernels::diffuse), DiffuseParameters>(kernels::diffuse);
			try_launch.template operator()<decltype(kernels::conductor), ConductorParameters>(kernels::conductor);
			try_launch.template operator()<decltype(kernels::dielectric), DielectricParameters>(kernels::dielectric);
			try_launch.template operator()<decltype(kernels::emissive), EmissiveParameters>(kernels::emissive);
		}

		Launcher(kernels::escaped, Capacity).launch(escape_packets, paths);
		material_queries.clear_async();
		escape_packets.clear_async();

		for (auto& indices : material_indices) indices.clear_async();
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
