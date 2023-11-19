#pragma once

#include <cassert>
#include <cuda_runtime.h>

#define HOST_DEVICE __host__ __device__

namespace cb
{

__host__
void cuda_check(cudaError error);

template<typename T>
__host__
void cuda_free_checked(T*& pointer)
{
	assert(pointer != nullptr);
	cuda_check(cudaFree(pointer));
	pointer = nullptr;
}

__device__
uint32_t get_thread_position1D();

__device__
UInt2 get_thread_position2D();

__device__
UInt3 get_thread_position3D();

} // cb
