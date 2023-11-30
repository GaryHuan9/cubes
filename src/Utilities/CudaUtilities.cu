#include "Utilities/CudaUtilities.hpp"
#include "Utilities/Vector.hpp"

#include <stdexcept>

namespace cb
{

__host__
void cuda_check(cudaError error)
{
	if (error == cudaError::cudaSuccess) return;
	std::string error_string(cudaGetErrorString(error));
	throw std::runtime_error("CUDA error: " + error_string);
}

__device__
uint32_t get_thread_index()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

}
