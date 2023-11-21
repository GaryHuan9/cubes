#pragma once

#include "main.hpp"
#include "CudaUtilities.hpp"

//Must be forward declared to allow inclusion of this file in non CUDA environments
template<typename T>
static inline T atomicAdd(T*, T);

namespace cb
{

template<typename T>
class CudaVector
{
public:
	class Accessor;
	using size_type = size_t;

	__host__
	explicit CudaVector(size_type capacity = 0) : max_count(capacity)
	{
		if (capacity == 0) return;
		cuda_check(cudaMalloc(&count, sizeof(size_type)));
		cuda_check(cudaMalloc(&pointer, capacity * sizeof(T)));
		clear();
	}

	CudaVector(CudaVector&& source) noexcept
	{
		*this = std::move(source);
	}

	CudaVector& operator=(CudaVector&& source) noexcept
	{
		if (max_count > 0)
		{
			cuda_free_checked(count);
			cuda_free_checked(pointer);
		}

		count = source.count;
		max_count = source.max_count;
		pointer = source.pointer;

		source.count = 0;
		source.length = 0;
		source.pointer = nullptr;
		return *this;
	}

	~CudaVector()
	{
		if (max_count == 0) return;
		cuda_free_checked(count);
		cuda_free_checked(pointer);
	}

	CudaVector(const CudaVector&) = delete;
	CudaVector& operator=(const CudaVector&) = delete;

	[[nodiscard]]
	size_type size() const
	{
		size_type result;
		cuda_check(cudaMemcpy(&result, count, sizeof(size_type), cudaMemcpyDefault));
		return result;
	}

	[[nodiscard]]
	size_type capacity() const { return max_count; }

	[[nodiscard]]
	const T* data() const { return pointer; }

	void clear()
	{
		cuda_check(cudaMemset(&count, 0, sizeof(size_type)));
	}

	operator Accessor() const { return CudaVector<T>::Accessor(*this); } // NOLINT(*-explicit-constructor)

private:
	size_type* count = nullptr;
	size_type max_count = 0;
	T* pointer = nullptr;
};

template<typename T>
class CudaVector<T>::Accessor
{
public:
	explicit Accessor(const CudaVector& source) : count(source.count), max_count(source.max_count), pointer(source.pointer) {}

	[[nodiscard]]
	__device__ size_type size() const { return *count; }

	[[nodiscard]]
	__device__ size_type capacity() const { return max_count; }

	__device__
	const T& operator[](size_type index) const
	{
		cuda_assert(index < size());
		return pointer[index];
	}

	__device__
	T& operator[](size_type index)
	{
		cuda_assert(index < size());
		return pointer[index];
	}

	__device__
	T& push_back(const T& value) { return emplace_back(value); }

	__device__
	T& push_back(T&& value) { return emplace_back(cuda_move(value)); }

	template<typename... Arguments>
	__device__
	T& emplace_back(Arguments&& ... arguments)
	{
		size_type index = atomicAdd(count, size_type(1));
		cuda_assert(index < capacity());
		return *new(pointer + index) T(cuda_forward<Arguments>(arguments)...);
	}

private:
	size_type* count = nullptr;
	size_type max_count = 0;
	T* pointer = nullptr;
};

} // cb
