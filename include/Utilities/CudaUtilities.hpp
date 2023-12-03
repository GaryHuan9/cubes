#pragma once

#include "main.hpp"

#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>

#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_NODISCARD [[nodiscard]] __host__ __device__

namespace cb
{

#if __NVCC__

__device__
inline uint32_t get_thread_index()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

#endif

__host__
inline void cuda_check(cudaError error)
{
	if (error == cudaError::cudaSuccess) return;
	std::string error_string(cudaGetErrorString(error));
	throw std::runtime_error("CUDA error: " + error_string);
}

template<typename T>
__host__
inline void cuda_malloc(T*& pointer, size_t count = 1)
{
	assert(pointer == nullptr);
#if NDEBUG
	cuda_check(cudaMalloc(&pointer, count * sizeof(T)));
#else
	cuda_check(cudaMallocManaged(&pointer, count * sizeof(T)));
#endif
}

template<typename T>
__host__
inline void cuda_free_checked(T*& pointer)
{
	assert(pointer != nullptr);
	cuda_check(cudaFree(pointer));
	pointer = nullptr;
}

template<typename T>
__host__
inline void cuda_copy(T* destination, const T* source)
{
	cuda_check(cudaMemcpy(destination, source, sizeof(T), cudaMemcpyDefault));
}

//These following four functions are simple wrappers to satisfy CLion's intellisense
//HOST_DEVICE
//inline void assert(bool value)
//{
//	assert(value);
//}

template<class T>
HOST_DEVICE
inline std::remove_reference_t<T>&& cuda_move(T&& value) noexcept
{
	return static_cast<typename std::remove_reference_t<T>&&>(value);
}

template<class T>
HOST_DEVICE
inline constexpr T&& cuda_forward(std::remove_reference_t<T>& value) noexcept
{
	return static_cast<T&&>(value);
}

template<class T>
HOST_DEVICE
inline constexpr T&& cuda_forward(std::remove_reference_t<T>&& value) noexcept
{
	return static_cast<T&&>(value);
}

}
