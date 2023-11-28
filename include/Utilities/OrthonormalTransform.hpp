#pragma once

#include "main.hpp"
#include "Math.hpp"
#include "Vector.hpp"
#include "CudaUtilities.hpp"

namespace cb
{

class OrthonormalTransform
{
public:
	HOST_DEVICE
	explicit OrthonormalTransform(const Float3& axis_z) : axis_z(axis_z)
	{
		assert(almost_one(axis_z.squared_magnitude()));

		if (!almost_zero(axis_z.x()) || !almost_zero(axis_z.y()))
		{
			axis_x = Float3(axis_z.y(), -axis_z.x(), 0.0f).normalized(); //Equivalent to cross(axis_z, Float3(0.0f, 0.0f, 1.0f))
			axis_y = cross(axis_z, axis_x);
		}
		else
		{
			axis_x = Float3(1.0f, 0.0f, 0.0f);
			axis_y = Float3(0.0f, axis_z.z(), -axis_z.y()).normalized(); //Equivalent to cross(axis_z, axis_x);
		}

		assert(almost_one(axis_x.squared_magnitude()));
		assert(almost_one(axis_y.squared_magnitude()));
	}

	HOST_DEVICE
	OrthonormalTransform(const Float3& axis_z, const Float3& axis_x) : axis_z(axis_z), axis_x(axis_x)
	{
		assert(almost_one(axis_z.squared_magnitude()));
		assert(almost_one(axis_x.squared_magnitude()));
		assert(almost_zero(axis_x.dot(axis_z)));

		axis_y = cross(axis_z, axis_x);
		assert(almost_one(axis_y.squared_magnitude()));
	}

	HOST_DEVICE_NODISCARD
	Float3 apply_forward(const Float3& direction) const
	{
		return Float3(axis_x.x() * direction.x() + axis_y.x() * direction.y() + axis_z.x() * direction.z(),
		              axis_x.y() * direction.x() + axis_y.y() * direction.y() + axis_z.y() * direction.z(),
		              axis_x.z() * direction.x() + axis_y.z() * direction.y() + axis_z.z() * direction.z());
	}

	HOST_DEVICE_NODISCARD
	Float3 apply_inverse(const Float3& direction) const
	{
		return Float3(axis_x.x() * direction.x() + axis_x.y() * direction.y() + axis_x.z() * direction.z(),
		              axis_y.x() * direction.x() + axis_y.y() * direction.y() + axis_y.z() * direction.z(),
		              axis_z.x() * direction.x() + axis_z.y() * direction.y() + axis_z.z() * direction.z());
	}

private:
	Float3 axis_x;
	Float3 axis_y;
	Float3 axis_z;
};

}
