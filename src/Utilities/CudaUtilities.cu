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
uint32_t get_thread_index1D()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
UInt2 get_thread_index2D()
{
	UInt2 thread_index(threadIdx.x, threadIdx.y);
	UInt2 block_index(blockIdx.x, blockIdx.y);
	UInt2 block_size(blockDim.x, blockDim.y);
	return block_index * block_size + thread_index;
}

__device__
UInt3 get_thread_index3D()
{
	UInt3 thread_index(threadIdx.x, threadIdx.y, threadIdx.z);
	UInt3 block_index(blockIdx.x, blockIdx.y, blockIdx.z);
	UInt3 block_size(blockDim.x, blockDim.y, blockDim.z);
	return block_index * block_size + thread_index;
}

} // cb
