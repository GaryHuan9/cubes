#include "Scenic/Camera.hpp"
#include "Rendering/Ray.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaUtilities.hpp"

#include <numbers>

namespace cb
{

static float to_radians(float angle)
{
	return angle * (std::numbers::pi_v<float> / 180.0f);
}

static float to_degrees(float angle)
{
	return angle * (180.0f / std::numbers::pi_v<float>);
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

Float3 Camera::get_position() const
{
	return position;
}

Float2 Camera::get_rotation() const
{
	return Float2(to_degrees(rotation.x()), to_degrees(rotation.y()));
}

void Camera::set_position(const Float3& new_position)
{
	position = new_position;
}

void Camera::set_rotation(const Float2& new_rotation)
{
	float x = std::min(std::max(new_rotation.x(), -90.0f), 90.0f);
	float y = new_rotation.y();

	while (y < -180.0f) y += 360.0f;
	while (y >= 180.0f) y -= 360.0f;

	rotation = Float2(to_radians(x), to_radians(y));
}

void Camera::set_field_of_view(float new_field_of_view)
{
	field_of_view = new_field_of_view;
	forward_distance = 0.5f / std::tanf(to_radians(field_of_view / 2.0f));
}

void Camera::move_position(const Float3& local_delta)
{
	Float2 delta(local_delta.x(), local_delta.z());
	delta = rotate(delta, -rotation.y());

	set_position(position + Float3(delta.x(), local_delta.y(), delta.y()));
}

HOST_DEVICE
Ray Camera::get_ray(Float2 uv)
{
	Float3 direction(uv.x(), uv.y(), forward_distance);
	direction = rotate(direction, rotation).normalized();
	return { position, direction };
}

} // cb
