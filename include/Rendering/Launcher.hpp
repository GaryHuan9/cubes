#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaUtilities.hpp"

#include <unordered_map>
#include <typeindex>

namespace cb
{

template<typename Kernel>
class Launcher
{
public:
	explicit Launcher(const Kernel& kernel, uint32_t region = 1) : kernel(kernel), block_count(region), block_size(1)
	{
		if (region <= 1) return;
		block_size = get_block_size(kernel);
		block_count = (region + block_size - 1) / block_size;
	}

#if __NVCC__

	template<typename... Arguments>
	void launch(const Arguments& ... arguments)
	{
		kernel<<<block_count, block_size>>>(arguments...);
	}

#endif

private:
	static uint32_t get_block_size(const Kernel& kernel)
	{
		if (best_block_size > 0) return best_block_size;

		int min_grid_size, block_size;
		cuda_check(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel));

		best_block_size = static_cast<uint32_t>(min_grid_size);
		return best_block_size;
	}

	const Kernel& kernel;
	uint32_t block_count;
	uint32_t block_size;

	inline static uint32_t best_block_size;
};

}
