#include "Rendering/Camera.hpp"
#include "Rendering/Ray.hpp"
#include "Utilities/CudaUtilities.hpp"

namespace cb
{

__device__
static Float2 rotate(const Float2& direction, float angle)
{
	float sin = sinf(angle);
	float cos = cosf(angle);

	return Float2(direction.x() * cos - direction.y() * sin,
	              direction.x() * sin + direction.y() * cos);
}

__device__
static Float3 rotate(const Float3& direction, const Float2& rotation)
{
	Float2 yz = rotate(Float2(direction.y(), direction.z()), rotation.x());
	Float2 xz = rotate(Float2(direction.x(), yz.y()), -rotation.y());
	return Float3(xz.x(), yz.x(), xz.y());
}

Ray Camera::get_ray(Float2 uv) const
{
	Float3 direction(uv.x(), uv.y(), forward_distance);
	direction = rotate(direction, rotation).normalized();
	return { position, direction };
}

} // cb
