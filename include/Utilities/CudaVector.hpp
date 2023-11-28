#pragma once

#include "main.hpp"
#include "CudaArray.hpp"
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
	explicit CudaVector(size_type capacity = 0) : array(capacity)
	{
		if (capacity == 0) return;
		cuda_check(cudaMalloc(&count, sizeof(size_type)));
		clear();
	}

	CudaVector(CudaVector&& source) noexcept
	{
		*this = std::move(source);
	}

	CudaVector& operator=(CudaVector&& source) noexcept
	{
		if (capacity() > 0) cuda_free_checked(count);

		count = source.count;
		array = std::move(source.array);

		source.count = 0;
		return *this;
	}

	~CudaVector()
	{
		if (capacity() == 0) return;
		cuda_free_checked(count);
	}

	CudaVector(const CudaVector&) = delete;
	CudaVector& operator=(const CudaVector&) = delete;

	[[nodiscard]]
	size_type size() const
	{
		if (capacity() == 0) return size_type();

		size_type result;
		cuda_copy(&result, count);
		return result;
	}

	[[nodiscard]]
	size_type capacity() const { return array.size(); }

	[[nodiscard]]
	const T* data() const { return array.data(); }

	void clear()
	{
		if (capacity() == 0) return;
		cuda_check(cudaMemset(count, 0, sizeof(size_type)));
	}

	operator Accessor() const // NOLINT(*-explicit-constructor)
	{
		assert(capacity() > 0);
		return Accessor(count, array);
	}

private:
	size_type* count = nullptr;
	CudaArray<T> array;
};

template<typename T>
class CudaVector<T>::Accessor
{
public:
	[[nodiscard]]
	__device__ size_type size() const { return *count; }

	[[nodiscard]]
	__device__ size_type capacity() const { return array.size(); }

	__device__
	const T& operator[](size_type index) const
	{
		assert(index < size());
		return array[index];
	}

	__device__
	T& operator[](size_type index)
	{
		assert(index < size());
		return array[index];
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
		assert(index < capacity());
		return array.emplace(index, cuda_forward<Arguments>(arguments)...);
	}

private:
	Accessor(size_type* count, const typename CudaArray<T>::Accessor& array) : count(count), array(array)
	{
		assert(count != nullptr);
	}

	size_type* count = nullptr;
	CudaArray<T>::Accessor array;
	friend CudaVector<T>;
};

}
