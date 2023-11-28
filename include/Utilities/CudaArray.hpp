#pragma once

#include "main.hpp"
#include "CudaUtilities.hpp"

namespace cb
{

template<typename T>
class CudaArray
{
public:
	class Accessor;
	using size_type = size_t;

	explicit CudaArray(size_type size = 0) : length(size)
	{
		if (size == 0) return;
		cuda_check(cudaMalloc(&pointer, size * sizeof(T)));
	}

	CudaArray(CudaArray&& source) noexcept
	{
		*this = std::move(source);
	}

	CudaArray& operator=(CudaArray&& source) noexcept
	{
		if (length > 0) cuda_free_checked(pointer);

		length = source.length;
		pointer = source.pointer;
		source.length = 0;
		source.pointer = nullptr;
		return *this;
	}

	~CudaArray()
	{
		if (length == 0) return;
		cuda_free_checked(pointer);
	}

	CudaArray(const CudaArray&) = delete;
	CudaArray& operator=(const CudaArray&) = delete;

	[[nodiscard]]
	size_type size() const { return length; }

	[[nodiscard]]
	const T* data() const { return pointer; }

	void clear()
	{
		cuda_check(cudaMemset(pointer, 0, size() * sizeof(T)));
	}

	operator Accessor() const // NOLINT(*-explicit-constructor)
	{
		assert(pointer != nullptr);
		return Accessor(length, pointer);
	}

private:
	size_type length = 0;
	T* pointer = nullptr;
};

template<typename T>
class CudaArray<T>::Accessor
{
public:
	[[nodiscard]]
	__device__ size_type size() const { return length; }

	__device__
	const T& operator[](size_type index) const
	{
		assert(index < size());
		return pointer[index];
	}

	__device__
	T& operator[](size_type index)
	{
		assert(index < size());
		return pointer[index];
	}

	template<typename... Arguments>
	__device__
	T& emplace(size_type index, Arguments&& ... arguments)
	{
		assert(index < size());
		return *new(pointer + index) T(cuda_forward<Arguments>(arguments)...);
	}

private:
	Accessor(size_type length, T* pointer) : length(length), pointer(pointer)
	{
		assert(pointer != nullptr);
	}

	size_type length = 0;
	T* pointer = nullptr;
	friend CudaArray<T>;
};

}
