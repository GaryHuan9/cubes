#include "Rendering/Kernels.hpp"
#include "Rendering/Packets.hpp"
#include "Rendering/Accumulator.hpp"
#include "Scenic/Camera.hpp"
#include "Scenic/Scene.hpp"

namespace cb::kernels
{

__global__
void render_begin(UInt2 resolution, uint32_t offset, uint32_t length, Array<NewPathPackets> results)
{
	uint32_t index = get_thread_index1D();
	if (index >= length) return;

	uint32_t position = index + offset;
	uint32_t y = position / resolution.x();
	uint32_t x = position - y * resolution.x();

	if (y >= resolution.y()) y = y % resolution.y();
	results.emplace(index, UInt2(x, y));
}

__global__
void new_path(UInt2 resolution, Camera* camera, const Array<NewPathPackets> packets, List<TracePackets> results)
{
	uint32_t index = get_thread_index1D();
	if (index >= packets.size()) return;
	const auto& packet = packets[index];

	Float2 uv = Float2(packet.pixel) - Float2(resolution) * 0.5f;
	float multiplier = 1.0f / static_cast<float>(resolution.x());
	results.emplace_back(packet.pixel, camera->get_ray(uv * multiplier));
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
void trace_rays(const List<TracePackets> packets, Array<HitPacket> results)
{
	uint32_t index = get_thread_index1D();
	if (index >= packets.size()) return;
	const auto& packet = packets[index];

	float distance;
	Float3 normal;

	if (!try_intersect_scene(packet.ray, distance, normal)) results.emplace(index);
	else results.emplace(index, distance, normal);
}

__global__
void render_end(UInt2 resolution, const List<TracePackets> packets, const Array<HitPacket> results, Array<Accumulator> accumulators)
{
	uint32_t index = get_thread_index1D();
	if (index >= packets.size()) return;
	const auto& packet = packets[index];
	const auto& result = results[index];

	Float3 color;

	if (result.hit())
	{
		color = result.normal * 0.5f + Float3(0.5f);
		//		color = Float3(max(result.normal.dot(Float3(1.0f).normalized()), 0.0f));
	}

	accumulators[packet.pixel.y() * resolution.x() + packet.pixel.x()].insert(color);
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

} // cb::kernel
