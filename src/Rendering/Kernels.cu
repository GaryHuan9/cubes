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
	uint32_t thread_index = get_thread_index();
	if (thread_index >= randoms.size()) return;
	curand_init(42, thread_index, 0, &randoms[thread_index]);
}

__global__
void new_path(Array<Path> paths, UInt2 resolution, uint32_t index_start, Array<curandState> randoms, Camera* camera, List<TraceQuery> trace_queries)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= paths.size()) return;

	uint32_t bounds = resolution.product();
	uint32_t result_index = index_start + thread_index;
	while (result_index >= bounds) result_index -= bounds;

	uint32_t y = result_index / resolution.x();
	uint32_t x = result_index - y * resolution.x();
	assert(UInt2(x, y) < resolution);

	curandState* random = &randoms[thread_index];

	Float2 offset = Float2(curand_uniform(random), curand_uniform(random));
	Float2 uv = Float2(UInt2(x, y)) + offset - Float2(resolution) * 0.5f;
	float multiplier = 1.0f / static_cast<float>(resolution.x());
	Ray ray = camera->get_ray(uv * multiplier);

	paths.emplace(thread_index);
	trace_queries.emplace_back(thread_index, ray);
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
	float extend = sqrt(extend2);
	distance = center - extend;

	if (distance < 0.0f) distance = center + extend;
	if (distance < 0.0f) return false;

	return true;
}

__device__
static bool try_intersect_voxels(const Ray& ray, float& distance, uint32_t& material, Float3& normal)
{
	__device__ static constexpr float VoxelSize = 0.1f;
	__device__ static constexpr float VoxelSizeR = 1.0f / VoxelSize;
	__device__ static constexpr Int3 GridSize = Int3(16 * 16);

	__device__ static constexpr Float3 Center = Float3(0.0f, 12.8f, 0.0f);
	__device__ static constexpr Float3 Extend = Float3(GridSize) / 2.0f;
	__device__ static constexpr Float3 Offset = Center - Float3(GridSize) / 2.0f * VoxelSize;

	Float3 origin = (ray.origin - Offset) * VoxelSizeR;
	Float3 direction_r = Float3(1.0f) / ray.direction;

	//Try to intersect with bounding box
	Float3 lengths0 = (Float3(0.0f) - origin) * direction_r;
	Float3 lengths1 = (Extend * 2.0f - origin) * direction_r;
	Float3 lengths_min = lengths0.min(lengths1);
	Float3 lengths_max = lengths0.max(lengths1);

	float near = max(lengths_min.x(), max(lengths_min.y(), lengths_min.z()));
	float far = min(lengths_max.x(), min(lengths_max.y(), lengths_max.z()));

	if (far < near || far < 0.0f) return false;
	distance = near;

	Int3 position(origin);

	if (distance > 0.0f)
	{
		Float3 point = origin + ray.direction * distance;
		position = Int3(point).max(Int3(0)).min(GridSize - Int3(1));

		if (lengths_min.x() > lengths_min.y())
		{
			if (lengths_min.x() > lengths_min.z()) position.x() += ray.direction.x() > 0.0f ? -1 : 1;
			else position.z() += ray.direction.z() > 0.0f ? -1 : 1;
		}
		else
		{
			if (lengths_min.y() > lengths_min.z()) position.y() += ray.direction.y() > 0.0f ? -1 : 1;
			else position.z() += ray.direction.z() > 0.0f ? -1 : 1;
		}
	}

	Int3 step = Int3(ray.direction.x() > 0.0f ? 1 : 0,
	                 ray.direction.y() > 0.0f ? 1 : 0,
	                 ray.direction.z() > 0.0f ? 1 : 0);

	Float3 max_lengths = (Float3(position + step) - origin) * direction_r;
	Float3 delta = direction_r.max(-direction_r);
	step = step * 2 - Int3(1);

	while (true)
	{
		normal = {};

		if (max_lengths.x() < max_lengths.y())
		{
			if (max_lengths.x() < max_lengths.z())
			{
				distance = max_lengths.at<0>();
				normal.at<0>() = -static_cast<float>(step.at<0>());
				max_lengths.at<0>() += delta.at<0>();
				position.at<0>() += step.at<0>();
			}
			else
			{
				distance = max_lengths.at<2>();
				normal.at<2>() = -static_cast<float>(step.at<2>());
				max_lengths.at<2>() += delta.at<2>();
				position.at<2>() += step.at<2>();
			}
		}
		else
		{
			if (max_lengths.y() < max_lengths.z())
			{
				distance = max_lengths.at<1>();
				normal.at<1>() = -static_cast<float>(step.at<1>());
				max_lengths.at<1>() += delta.at<1>();
				position.at<1>() += step.at<1>();
			}
			else
			{
				distance = max_lengths.at<2>();
				normal.at<2>() = -static_cast<float>(step.at<2>());
				max_lengths.at<2>() += delta.at<2>();
				position.at<2>() += step.at<2>();
			}
		}

		if (!(Int3(0) <= position && position < GridSize)) return false;

		bool has_voxel = true;

		auto check = []HOST_DEVICE(int32_t value, uint32_t level)
		{
			value = (value >> (level * 2)) & 0b11;
			return (value == 0 || value == 3) ? 1 : 0;
		};

		for (uint32_t level = 0; level < 4; ++level)
		{
			uint32_t edge = 0;
			edge += check(position.x(), level);
			edge += check(position.y(), level);
			edge += check(position.z(), level);
			has_voxel &= edge >= 2;
		}

		if (has_voxel)
		{
			distance *= VoxelSize;
			material = 0;
			return true;
		}
	}
}

__device__
static bool try_intersect_scene(const Ray& ray, float& distance, uint32_t& material, Float3& normal)
{
	constexpr size_t Count = 4;
	__constant__ static const Float4 Spheres[Count] = { Float4(0.0f, 1.0f, 0.0f, 1.0f),
	                                                    Float4(1.0f, 2.0f, 2.0f, 0.5f),
	                                                    Float4(-1.0f, 3.0f, 1.0f, 0.5f),
	                                                    Float4(0.0f, -100.0f, 0.0f, 100.0f) };

	__constant__ static const uint32_t Materials[Count] = { 3, 1, 3, 0 };

	distance = INFINITY;
	if (!try_intersect_voxels(ray, distance, material, normal)) distance = INFINITY;

	if (ray.direction.y() < 0.0f)
	{
		float new_distance = ray.origin.y() / -ray.direction.y();
		if (new_distance < distance && new_distance >= 0.0f)
		{
			distance = new_distance;
			material = 1;
			normal = Float3(0.0f, 1.0f, 0.0f);
		}
	}

	for (size_t i = 0; i < 0; ++i)
	{
		const Float4& sphere = Spheres[i];
		Float3 position(sphere.x(), sphere.y(), sphere.z());
		float radius(sphere.w());
		float new_distance;

		if (try_intersect_sphere(position, radius, ray, new_distance) && new_distance < distance)
		{
			distance = new_distance;
			material = Materials[i];
			normal = ray.get_point(distance) - position;
		}
	}

	if (isinf(distance)) return false;
	normal = normal.normalized();
	return true;
}

__global__
void trace(const List<TraceQuery> trace_queries, List<MaterialQuery> material_queries, List<EscapedPacket> escaped_packets)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= trace_queries.size()) return;
	const auto& query = trace_queries[thread_index];

	float distance;
	uint32_t material;
	Float3 normal;

	if (!try_intersect_scene(query.ray, distance, material, normal))
	{
		escaped_packets.emplace_back(query.path_index, query.ray.direction);
		return;
	}

	material_queries.emplace_back(query, distance, material, normal);
}

__global__
void pre_material(List<MaterialQuery> material_queries, Array<List<uint32_t>> material_indices, Array<curandState> randoms)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= material_queries.size()) return;
	auto& query = material_queries[thread_index];
	curandState* random = &randoms[thread_index];

	material_indices[query.material].emplace_back(thread_index);
	query.initialize(Float2(curand_uniform(random), curand_uniform(random)));
}

HOST_DEVICE
static Float2 project_disk(float radius, float angle)
{
	Float2 disk;
	sincos(angle, &disk.y(), &disk.x());
	return disk * radius;
}

HOST_DEVICE
static Float3 cosine_hemisphere(const Float2& sample)
{
	constexpr float Tau = std::numbers::pi_v<float> * 2.0f;
	float radius = sqrt(sample.x());
	float angle = Tau * sample.y();

	Float2 disk = project_disk(radius, angle);
	float z = 1.0f - disk.squared_magnitude();
	return Float3(disk.x(), disk.y(), sqrt0(z));
}

HOST_DEVICE
static Float3 uniform_sphere(const Float2& sample)
{
	constexpr float Tau = std::numbers::pi_v<float> * 2.0f;
	float z = sample.x() * 2.0f - 1.0f;
	float radius = sqrtf(1.0f - z * z);
	float angle = Tau * sample.y();

	Float2 disk = project_disk(radius, angle);
	return Float3(disk.x(), disk.y(), z);
}

HOST_DEVICE
static float cos_phi(const Float3& direction) { return direction.z(); }

HOST_DEVICE
static void make_same_side(const Float3& outgoing, Float3& incident)
{
	if (outgoing.z() * incident.z() >= 0.0f) return;
	incident.z() = -incident.z();
}

__device__
static void advance(const MaterialQuery& query, const Float3& incident, const Float3& scatter, Array<Path> paths, List<TraceQuery> trace_queries)
{
	assert(almost_one(incident.squared_magnitude()));

	auto& path = paths[query.path_index];
	//	float cos = abs(cosine_phi(incident)); //Currently disabled because both materials don't need it
	if (!path.bounce(scatter/* * cos*/)) return;

	Float3 incident_world = query.transform().apply_forward(incident);
	Ray ray(query.point + incident_world * 1E-3f, incident_world);
	//	Ray ray(query.point + query.normal * 1E-4f, incident_world);
	trace_queries.emplace_back(query.path_index, ray);
}

__global__
void diffuse(const List<uint32_t> material_indices, const List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, const DiffuseParameters parameters)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= material_indices.size()) return;
	uint32_t index = material_indices[thread_index];
	const auto& query = material_queries[index];

	Float3 incident = cosine_hemisphere(query.get_sample());
	make_same_side(query.get_outgoing(), incident);
	advance(query, incident, parameters.albedo, paths, trace_queries);
}

__global__
void conductor(const List<uint32_t> material_indices, const List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, const ConductorParameters parameters)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= material_indices.size()) return;
	uint32_t index = material_indices[thread_index];
	const auto& query = material_queries[index];

	Float3 incident = -query.get_outgoing();

	if (!almost_zero(parameters.roughness))
	{
		Float3 sphere = uniform_sphere(query.get_sample());
		incident += sphere * parameters.roughness;
		incident = incident.normalized();
	}

	make_same_side(query.get_outgoing(), incident);
	advance(query, incident, parameters.albedo, paths, trace_queries);
}

HOST_DEVICE
static float fresnel_schlick(float eta_outgoing, float eta_incident, float cos_outgoing, float)
{
	float normal_reflectance = (eta_outgoing - eta_incident) / (eta_outgoing + eta_incident);
	normal_reflectance *= normal_reflectance;

	float cos = 1.0f - abs(cos_outgoing);
	float cos2 = cos * cos;
	float cos5 = cos2 * cos2 * cos;
	return normal_reflectance + (1.0f - normal_reflectance) * cos5;
}

HOST_DEVICE
static float fresnel(float eta_outgoing, float eta_incident, float cos_outgoing, float cos_incident)
{
	float cos_o = abs(cos_outgoing);
	float cos_i = abs(cos_incident);

	float para0 = eta_incident * cos_o;
	float para1 = eta_outgoing * cos_i;
	float perp0 = eta_outgoing * cos_o;
	float perp1 = eta_incident * cos_i;

	float para = (para0 - para1) / (para0 + para1);
	float perp = (perp0 - perp1) / (perp0 + perp1);
	return (para * para + perp * perp) / 2.0f;
}

HOST_DEVICE
static float get_cos_incident(float eta, float cos_outgoing)
{
	float sin_outgoing2 = 1.0f - cos_outgoing * cos_outgoing;
	float sin_incident2 = eta * eta * sin_outgoing2;
	if (sin_incident2 >= 1.0f) return 0.0f;
	return sqrt0(1.0f - sin_incident2);
}

__global__
void dielectric(const List<uint32_t> material_indices, List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, const DielectricParameters parameters)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= material_indices.size()) return;
	uint32_t index = material_indices[thread_index];
	const auto& query = material_queries[index];

	//Defaults to specular reflection
	Float3 outgoing = query.get_outgoing();
	Float3 incident = -outgoing;
	make_same_side(outgoing, incident);

	//Calculate eta
	float cos_o = cos_phi(outgoing);
	float eta_outgoing = 1.0f; //eta above surface
	float eta_incident = parameters.refractive_index;
	if (cos_o < 0.0f) cuda_swap(eta_outgoing, eta_incident);

	float eta = eta_outgoing / eta_incident;
	float cos_i = get_cos_incident(eta, cos_o);

	//Evaluate fresnel
	float evaluated = fresnel(eta_outgoing, eta_incident, cos_o, cos_i);
	if (almost_zero(cos_i)) evaluated = 1.0f; //Total internal reflection

	if (query.get_sample().x() > evaluated)
	{
		//Specular transmission
		float z = eta * cos_o + (cos_o < 0.0f ? cos_i : -cos_i);
		incident = Float3(0.0f, 0.0f, z) - outgoing * eta;
		incident = incident.normalized();
	}

	advance(query, incident, parameters.albedo, paths, trace_queries);
}

__global__
void emissive(const List<uint32_t> material_indices, List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, const EmissiveParameters parameters)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= material_indices.size()) return;
	uint32_t index = material_indices[thread_index];
	const auto& query = material_queries[index];

	if (cos_phi(query.get_outgoing()) < 0.0f) return;
	auto& path = paths[query.path_index];
	path.contribute(parameters.albedo);
}

__global__
void escaped(const List<EscapedPacket> escaped_packets, Array<Path> paths)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= escaped_packets.size()) return;
	const auto& packet = escaped_packets[thread_index];
	auto& path = paths[packet.path_index];

	//	path.contribute(Float3(0.1f));
	path.contribute(packet.direction * packet.direction);
}

__global__
void accumulate(const Array<Path> paths, uint32_t index_start, Array<Accumulator> accumulators)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= paths.size()) return;
	const auto& path = paths[thread_index];

	uint32_t result_index = index_start + thread_index;
	uint32_t count = accumulators.size();
	while (result_index >= count) result_index -= count;
	accumulators[result_index].insert(path.get_result());
}

__device__
static uint32_t convert(const Accumulator& accumulator)
{
	Float3 color = accumulator.current();
	auto convert = [] HOST_DEVICE(float value) { return min((uint32_t)(sqrt(value) * 256.0f), 255); };
	return 0xFF000000 | (convert(color.x()) << 16) | (convert(color.y()) << 8) | convert(color.z());
}

__global__
void output_surface(UInt2 resolution, Array<Accumulator> accumulators, cudaSurfaceObject_t surface)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= accumulators.size()) return;

	int x = static_cast<int>(thread_index % resolution.x() * sizeof(uint32_t));
	int y = static_cast<int>(resolution.y() - thread_index / resolution.x() - 1);
	surf2Dwrite(convert(accumulators[thread_index]), surface, x, y);
}

__global__
void output_buffer(UInt2 resolution, Array<Accumulator> accumulators, Array<uint32_t> array)
{
	uint32_t thread_index = get_thread_index();
	if (thread_index >= accumulators.size()) return;

	uint32_t x = thread_index % resolution.x();
	uint32_t y = thread_index / resolution.x();
	uint32_t index = (resolution.y() - y - 1) * resolution.x() + x;
	array[index] = convert(accumulators[thread_index]);
}

}
