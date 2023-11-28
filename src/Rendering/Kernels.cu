#include "Rendering/Kernels.hpp"
#include "Rendering/Structures.hpp"
#include "Utilities/OrthonormalTransform.hpp"
#include "Scenic/Camera.hpp"
#include "Scenic/Scene.hpp"

#include <numbers>

namespace cb::kernels
{

__global__
void new_random(Array<curandState> randoms)
{
	uint32_t index = get_thread_index1D();
	if (index >= randoms.size()) return;
	curand_init(42, index, 0, &randoms[index]);
}

__global__
void new_path(Array<Path> paths, UInt2 resolution, uint32_t index_start, Camera* camera, List<TraceQuery> trace_queries)
{
	uint32_t index = get_thread_index1D();
	if (index >= paths.size()) return;

	uint32_t result_index = index_start + index;
	uint32_t count = resolution.x() * resolution.y();
	while (result_index >= count) result_index -= count;

	uint32_t y = result_index / resolution.x();
	uint32_t x = result_index - y * resolution.x();
	assert(UInt2(x, y) < resolution);

	Float2 uv = Float2(UInt2(x, y)) - Float2(resolution) * 0.5f;
	float multiplier = 1.0f / static_cast<float>(resolution.x());
	Ray ray = camera->get_ray(uv * multiplier);

	paths.emplace(index, result_index);
	trace_queries.emplace_back(index, ray);
}

HOST_DEVICE
static bool try_intersect_sphere(const Float3& position, float radius, const Ray& ray, float& distance)
{
	//Test ray direction
	Float3 offset = ray.origin - position;
	float radius2 = radius * radius;
	float center = -offset.dot(ray.direction);

	float extend2 = center * center + radius2 - offset.squared_magnitude();

	if (extend2 < 0.0f) return false;

	//Find appropriate distance
	float extend = sqrtf(extend2);
	distance = center - extend;

	if (distance < 0.0f) distance = center + extend;
	if (distance < 0.0f) return false;

	return true;
}

__device__
static bool try_intersect_scene(const Ray& ray, float& distance, Float3& normal)
{
	constexpr size_t Count = 4;
	__constant__ static const Float4 Spheres[Count] = { Float4(0.0f, 1.0f, 0.0f, 1.0f),
	                                                    Float4(1.0f, 2.0f, 2.0f, 0.5f),
	                                                    Float4(-1.0f, 3.0f, 1.0f, 0.5f),
	                                                    Float4(0.0f, -100.0f, 0.0f, 100.0f) };

	distance = INFINITY;

	for (const Float4& sphere : Spheres)
	{
		Float3 position(sphere.x(), sphere.y(), sphere.z());
		float radius(sphere.w());
		float new_distance;

		if (try_intersect_sphere(position, radius, ray, new_distance) && new_distance < distance)
		{
			distance = new_distance;
			normal = ray.get_point(distance) - position;
		}
	}

	if (isinf(distance)) return false;
	normal = normal.normalized();
	return true;
}

__global__
void trace(List<TraceQuery> trace_queries)
{
	uint32_t index = get_thread_index1D();
	if (index >= trace_queries.size()) return;
	auto& query = trace_queries[index];

	float distance;
	uint32_t material = 0;
	Float3 normal;

	if (!try_intersect_scene(query.ray, distance, normal)) return;
	query.try_record(distance, material, normal);
}

__global__
void shade(const List<TraceQuery> trace_queries, List<MaterialQuery> material_queries, List<EscapedPacket> escaped_packets, Array<curandState> randoms)
{
	uint32_t index = get_thread_index1D();
	if (index >= trace_queries.size()) return;
	const auto& query = trace_queries[index];

	if (query.hit())
	{
		curandState* random = &randoms[index];
		Float2 sample(curand_uniform(random), curand_uniform(random));
		material_queries.emplace_back(query, sample);

		//TODO find material queue
	}
	else escaped_packets.emplace_back(query.path_index, query.ray.direction);
}

HOST_DEVICE
static Float3 cosine_hemisphere(const Float2& sample)
{
	constexpr float Tau = std::numbers::pi_v<float> * 2.0f;
	float radius = sqrtf(sample.x());
	float angle = Tau * sample.y();

	Float2 disk;
	sincosf(angle, &disk.y(), &disk.x());
	disk *= radius;

	float z = 1.0f - disk.squared_magnitude();
	return Float3(disk.x(), disk.y(), sqrtf(z));
}

__global__
void diffuse(List<MaterialQuery> material_queries)
{
	uint32_t index = get_thread_index1D();
	if (index >= material_queries.size()) return;
	auto& query = material_queries[index];

	Float3 incident = cosine_hemisphere(query.sample);
	constexpr float PiR = 1.0f / std::numbers::pi_v<float>;

	float pdf = incident.z() * PiR;
	if (query.get_outgoing().z() < 0.0f) incident.z() *= -1.0f;
	query.set_sampled(incident, Float3(0.8f) * PiR, pdf);
}

__global__
void advance(List<MaterialQuery> material_queries, List<TraceQuery> trace_queries, Array<Path> paths, Array<Accumulator> accumulators)
{
	uint32_t index = get_thread_index1D();
	if (index >= material_queries.size()) return;
	const auto& query = material_queries[index];
	auto& path = paths[query.path_index];

	float pdf = query.get_pdf();
	Float3 incident = query.get_incident_world().normalized();
	Float3 scatter = query.get_scatter() * incident.dot(query.normal) / pdf;

	if (!almost_zero(pdf) && path.bounce(scatter))
	{
		Ray ray(query.point + incident * 1E-3f, incident);
		trace_queries.emplace_back(query.path_index, ray);
	}
	else accumulators[path.result_index].insert(path.get_result());
}

__global__
void escaped(const List<EscapedPacket> escaped_packets, Array<Path> paths, Array<Accumulator> accumulators)
{
	uint32_t index = get_thread_index1D();
	if (index >= escaped_packets.size()) return;
	const auto& packet = escaped_packets[index];
	auto& path = paths[packet.path_index];

	//TODO fancier evaluate infinite
	path.contribute(Float3(0.1f));

	accumulators[path.result_index].insert(path.get_result());
}

__global__
void output(UInt2 resolution, Array<Accumulator> accumulators, cudaSurfaceObject_t surface)
{
	UInt2 position = get_thread_index2D();
	if (!(position < resolution)) return;

	Float3 color = accumulators[position.y() * resolution.x() + position.x()].current();
	auto convert = [] __device__(float value) { return min((uint32_t)(sqrtf(value) * 256.0f), 255); };
	uint32_t value = 0xFF000000 | (convert(color.x()) << 16) | (convert(color.y()) << 8) | convert(color.z());

	int x = static_cast<int>(position.x() * sizeof(uint32_t));
	int y = static_cast<int>(resolution.y() - position.y() - 1);
	surf2Dwrite(value, surface, x, y);
}

}
