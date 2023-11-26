#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"

namespace cb
{

class Camera
{
public:
	Camera()
	{
		set_position(Float3());
		set_rotation(Float2());
		set_field_of_view(65.0f);
	}

	void set_position(const Float3& new_position);
	void set_rotation(const Float2& new_rotation);
	void set_field_of_view(float new_field_of_view);

	[[nodiscard]]
	HOST_DEVICE
	Ray get_ray(Float2 uv);

private:
	Float3 position;
	Float2 rotation;
	float field_of_view{};
	float forward_distance{};
};

} // cb
