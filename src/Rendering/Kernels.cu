#include "Rendering/Kernels.hpp"
#include "Rendering/Packets.hpp"
#include "Rendering/Accumulator.hpp"
#include "Scenic/Camera.hpp"
#include "Scenic/Scene.hpp"

namespace cb::kernels
{

__global__
void test(UInt2 resolution, Array<Accumulator> accumulators, float time)
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
void render_begin(UInt2 resolution, Array<NewPathPackets> results)
{
	uint32_t index = get_thread_index1D();
	if (index >= results.size()) return;

	UInt2 position = UInt2(index % resolution.x(), index / resolution.y());
	results.emplace(index, position);
}

__global__
void new_path(UInt2 resolution, Camera* camera, const List<NewPathPackets> packets, List<TracePackets> results)
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

__global__
void trace_rays(const List<TracePackets> packets, Array<HitPacket> results)
{
	uint32_t index = get_thread_index1D();
	if (index >= packets.size()) return;
	const auto& packet = packets[index];

	Float3 position = Float3(0.0f, 0.0f, 0.0f);
	float radius = 1.0f;
	float distance;

	if (try_intersect_sphere(position, radius, packet.ray, distance))
	{
		Float3 normal = packet.ray.get_point(distance) - position;
		results.emplace(index, distance, normal.normalized());
	}
	else results.emplace(index);
}

__global__
void render_end(UInt2 resolution, const Array<TracePackets> packets, const Array<HitPacket> results, Array<Accumulator> accumulators)
{
	uint32_t index = get_thread_index1D();
	if (index >= packets.size()) return;
	const auto& packet = packets[index];
	const auto& result = results[index];

	Float3 color = result.hit() ? Float3(min(1.0f / result.distance, 1.0f)) : Float3();
	uint32_t destination = packet.pixel.y() * resolution.x() + packet.pixel.x();
	accumulators[destination].insert(color);
}

__global__
void output(UInt2 resolution, Array<Accumulator> accumulators, cudaSurfaceObject_t surface)
{
	UInt2 position = get_thread_index2D();
	if (!(position < resolution)) return;

	Float3 color = accumulators[position.y() * resolution.x() + position.x()].current();

	auto convert = [] __device__(float value) { return min((uint32_t)(sqrtf(value) * 256.0f), 255); };
	uint32_t value = 0xFF000000 | (convert(color.x()) << 16) | (convert(color.y()) << 8) | convert(color.z());
	surf2Dwrite(value, surface, static_cast<int>(position.x() * sizeof(uint32_t)), static_cast<int>(position.y()));
}

} // cb::kernel
