#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"

namespace cb
{

class Accumulator
{
public:
	[[nodiscard]]
	HOST_DEVICE
	Float3 current() const
	{
		if (count == 0) return {};
		return total / static_cast<float>(count);
	}

	[[nodiscard]]
	HOST_DEVICE
	uint32_t population() const
	{
		return count;
	}

	HOST_DEVICE
	void insert(const Float3& value)
	{
		//TODO: ensure compiler does not optimize error out
		Float3 delta = value - error;
		Float3 new_total = total + delta;
		error = new_total - total - delta;
		total = new_total;
		++count;
	}

private:
	Float3 total;
	Float3 error;
	uint32_t count{};
};

} // cb
