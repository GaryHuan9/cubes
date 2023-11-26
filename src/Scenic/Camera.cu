#include "Scenic/Camera.hpp"
#include "Rendering/Ray.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaUtilities.hpp"

#include <numbers>

namespace cb
{

void Camera::set_position(const Float3& new_position)
{
	position = new_position;
}

void Camera::set_rotation(const Float2& new_rotation)
{
	rotation = new_rotation;
}

void Camera::set_field_of_view(float new_field_of_view)
{
	field_of_view = new_field_of_view;
	float radians = field_of_view * std::numbers::pi_v<float> / 180.0f;
	forward_distance = 0.5f / std::tanf(radians / 2.0f);
}

HOST_DEVICE
static Float2 rotate(const Float2& direction, float angle)
{
	float sin = sinf(angle);
	float cos = cosf(angle);

	return Float2(direction.x() * cos - direction.y() * sin,
	              direction.x() * sin + direction.y() * cos);
}

HOST_DEVICE
static Float3 rotate(const Float3& direction, const Float2& rotation)
{
	Float2 yz = rotate(Float2(direction.y(), direction.z()), rotation.x());
	Float2 xz = rotate(Float2(direction.x(), yz.y()), -rotation.y());
	return Float3(xz.x(), yz.x(), xz.y());
}

HOST_DEVICE
Ray Camera::get_ray(Float2 uv)
{
	Float3 direction(uv.x(), uv.y(), forward_distance);
	direction = rotate(direction, rotation).normalized();
	return { position, direction };
}

} // cb
