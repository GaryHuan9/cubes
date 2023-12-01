#pragma once

#include "main.hpp"
#include "Utilities/OrthonormalTransform.hpp"

namespace cb
{

class Accumulator
{
public:
	[[nodiscard]]
	__device__
	Float3 current() const
	{
		if (count == 0) return {};
		//		return total / static_cast<float>(count);

		double multiplier = 1.0 / static_cast<double>(count);
		return Float3(x * multiplier, y * multiplier, z * multiplier);
	}

	[[nodiscard]]
	__device__
	uint32_t population() const
	{
		return count;
	}

	__device__
	void insert(const Float3& value)
	{
		//		Float3 delta = value - error;
		//		Float3 new_total = total + delta;
		//		error = new_total - total - delta;
		//		total = new_total;
		//		++count;

		atomicAdd(&x, value.x());
		atomicAdd(&y, value.y());
		atomicAdd(&z, value.z());
		atomicAdd(&count, 1);
	}

private:
	//	Float3 total;
	//	Float3 error;
	//	uint32_t count{};

	double x{};
	double y{};
	double z{};
	uint64_t count{};
};

class Ray
{
public:
	HOST_DEVICE
	Ray(const Float3& origin, const Float3& direction) : origin(origin), direction(direction) {}

	HOST_DEVICE_NODISCARD
	Float3 get_point(float distance) const
	{
		return origin + direction * distance;
	}

	const Float3 origin;
	const Float3 direction;
};

class Path
{
public:
	__device__
	Path() : energy(1.0f) {}

	HOST_DEVICE_NODISCARD
	Float3 get_result() const
	{
		return result;
	}

	HOST_DEVICE
	void contribute(const Float3& value)
	{
		result += energy * value;
	}

	HOST_DEVICE
	bool bounce(const Float3& value)
	{
		energy *= value;
		return positive(luminance(energy));
	}

private:
	Float3 result;
	Float3 energy;
};

class TraceQuery
{
public:
	__device__
	TraceQuery(size_t path_index, Ray ray) : path_index(path_index), ray(cuda_move(ray)) {}

	const size_t path_index;
	const Ray ray;
};

class EscapedPacket
{
public:
	__device__
	explicit EscapedPacket(size_t path_index, const Float3& direction) : path_index(path_index), direction(direction) {}

	const size_t path_index;
	const Float3 direction;
};

class MaterialQuery
{
public:
	__device__
	MaterialQuery(const TraceQuery& query, float distance, uint32_t material, const Float3& normal) :
		path_index(query.path_index), material(material), normal(normal),
		point(query.ray.get_point(distance)), outgoing(query.ray.direction) {}

	const size_t path_index;
	const uint32_t material;
	const Float3 normal;
	const Float3 point;

	HOST_DEVICE_NODISCARD
	OrthonormalTransform transform() const
	{
		return OrthonormalTransform(normal);
	}

	HOST_DEVICE_NODISCARD Float3 get_outgoing() const { return outgoing; }

	HOST_DEVICE_NODISCARD Float2 get_sample() const { return sample; }

	HOST_DEVICE
	void initialize(const Float2& new_sample)
	{
		outgoing = transform().apply_inverse(-outgoing);
		sample = new_sample;
	}

private:
	Float3 outgoing;
	Float2 sample;
};

class DiffuseParameters
{
public:
	HOST_DEVICE
	explicit DiffuseParameters(const Float3& albedo) : albedo(albedo) {}

	const Float3 albedo;
};

class ConductorParameters
{
public:
	HOST_DEVICE
	ConductorParameters(const Float3& albedo, float roughness) : albedo(albedo), roughness(roughness) {}

	const Float3 albedo;
	const float roughness;
};

class EmissiveParameters
{
public:
	HOST_DEVICE
	explicit EmissiveParameters(const Float3& albedo) : albedo(albedo) {}

	const Float3 albedo;
};

}
