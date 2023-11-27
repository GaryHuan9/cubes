#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"

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
		return total / static_cast<float>(count);
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
		atomicAdd(&total.x(), value.x());
		atomicAdd(&total.y(), value.y());
		atomicAdd(&total.z(), value.z());
		atomicAdd(&count, 1);

		//		Float3 delta = value - error;
		//		Float3 new_total = total + delta;
		//		error = new_total - total - delta;
		//		total = new_total;
		//		++count;
	}

private:
	Float3 total;
	//	Float3 error;
	uint32_t count{};
};

} // cb
