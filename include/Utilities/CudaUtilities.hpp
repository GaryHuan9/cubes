#pragma once

#include "main.hpp"

#include <cassert>
#include <type_traits>
#include <cuda_runtime.h>

#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_NODISCARD [[nodiscard]] __host__ __device__

namespace cb
{

__device__
uint32_t get_thread_index1D();

__device__
UInt2 get_thread_index2D();

__device__
UInt3 get_thread_index3D();

__host__
void cuda_check(cudaError error);

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
