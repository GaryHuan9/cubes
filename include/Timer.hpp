#pragma once

#include "main.hpp"

namespace cb
{

class Timer
{
public:
	[[nodiscard]]
	uint64_t frame_time() const
	{
		return last_delta_time;
	}

	[[nodiscard]]
	uint64_t time() const
	{
		return update_time;
	}

	[[nodiscard]]
	uint64_t frame_count() const
	{
		return update_count;
	}

	void update(uint64_t delta_time)
	{
		last_delta_time = delta_time;
		update_time += delta_time;
		++update_count;
	}

	static float as_float(uint64_t time)
	{
		return static_cast<float>(time) * 1.0E-6f;
	}

private:
	uint64_t last_delta_time{};
	uint64_t update_time{};
	uint64_t update_count{};
};

}
