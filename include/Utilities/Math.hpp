#pragma once

#include "main.hpp"
#include "Vector.hpp"
#include "CudaUtilities.hpp"

#include <numbers>

namespace cb
{

HOST_DEVICE
inline float to_radians(float angle)
{
	return angle * (std::numbers::pi_v<float> / 180.0f);
}

HOST_DEVICE
inline float to_degrees(float angle)
{
	return angle * (180.0f / std::numbers::pi_v<float>);
}

HOST_DEVICE
inline bool almost_zero(float value, float epsilon = 8E-7f)
{
	return (-epsilon < value) & (value < epsilon);
}

HOST_DEVICE
inline bool almost_one(float value, float epsilon = 3E-5f)
{
	return (1.0f - epsilon < value) & (value < 1.0f + epsilon);
}

HOST_DEVICE
inline bool positive(float value, float epsilon = 8E-7f)
{
	return value > epsilon;
}

HOST_DEVICE
inline float sqrt0(float value)
{
	return sqrt(max(value, 0.0f));
}

HOST_DEVICE
static float difference_of_products(float value0, float value1, float value2, float value3)
{
	float product = value2 * value3;
	float difference = fma(value0, value1, -product);
	float error = fma(-value2, value3, product);
	return difference + error;
}

HOST_DEVICE
inline Float3 cross(const Float3& value0, const Float3& value1)
{
	return Float3(difference_of_products(value0.y(), value1.z(), value0.z(), value1.y()),
	              difference_of_products(value0.z(), value1.x(), value0.x(), value1.z()),
	              difference_of_products(value0.x(), value1.y(), value0.y(), value1.x()));
}

HOST_DEVICE
inline Float3 reflect(const Float3& value, const Float3& normal)
{
	return normal * (2.0f * value.dot(normal)) - value;
}

HOST_DEVICE
inline float luminance(const Float3& value)
{
	return value.dot(Float3(0.212671f, 0.715160f, 0.072169f));
}

}
