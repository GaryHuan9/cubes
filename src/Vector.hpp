#pragma once

#include "main.hpp"
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cuda_runtime.h>

namespace cb
{

#define HOST_DEVICE_CONSTEXPR HOST_DEVICE_ENTRY constexpr

template<typename T, size_t D>
class Vector
{
private:
	using V = Vector<T, D>;
	using F = std::conditional_t<sizeof(T) <= sizeof(float), float, double>;

public:
	HOST_DEVICE_CONSTEXPR Vector() : field{}, inner{} {}

	HOST_DEVICE_CONSTEXPR Vector(const V& value) : field(value.field), inner(value.inner) {}

	HOST_DEVICE_CONSTEXPR Vector(T field, Vector<T, D - 1> inner) : field(field), inner(inner) {}

	HOST_DEVICE_CONSTEXPR explicit Vector(T value) : field(value), inner(value) {}

	template<typename... Ts, std::enable_if_t<sizeof...(Ts) == D - 1, bool> = true>
	HOST_DEVICE_CONSTEXPR explicit Vector(T value, Ts... arguments) :  field(value), inner(arguments...) {}

	template<class U>
	HOST_DEVICE_CONSTEXPR explicit Vector(U value) : Vector(static_cast<T>(value)) {}

	template<class U>
	HOST_DEVICE_CONSTEXPR explicit Vector(Vector<U, D> value) : Vector(value.template as<T>()) {}

	template<size_t I, std::enable_if_t<I < D, bool> = true>
	HOST_DEVICE_CONSTEXPR
	T& at()
	{
		if constexpr (I == 0) return field;
		else return inner.template at<I - 1>();
	}

	template<size_t I>
	HOST_DEVICE_CONSTEXPR T at() const { return const_cast<V*>(this)->template at<I>(); }

	HOST_DEVICE_CONSTEXPR T& x() { return at<0>(); }

	HOST_DEVICE_CONSTEXPR T& y() { return at<1>(); }

	HOST_DEVICE_CONSTEXPR T& z() { return at<2>(); }

	HOST_DEVICE_CONSTEXPR T& w() { return at<3>(); }

	HOST_DEVICE_CONSTEXPR T x() const { return at<0>(); }

	HOST_DEVICE_CONSTEXPR T y() const { return at<1>(); }

	HOST_DEVICE_CONSTEXPR T z() const { return at<2>(); }

	HOST_DEVICE_CONSTEXPR T w() const { return at<3>(); }

	template<class U>
	HOST_DEVICE_CONSTEXPR Vector<U, D> as() const { return Vector(static_cast<U>(field), inner.template as<U>()); }

	HOST_DEVICE_CONSTEXPR T dot(V value) const { return field * value.field + inner.dot(value.inner); }

	HOST_DEVICE_CONSTEXPR T squared_magnitude() const { return dot(*this); }

	HOST_DEVICE_CONSTEXPR F magnitude() const { return sqrt(static_cast<F>(squared_magnitude())); }

	HOST_DEVICE_CONSTEXPR Vector<F, D> normalized() const { return as<F>() * static_cast<F>(1.0) / sqrt(static_cast<F>(squared_magnitude())); }

	HOST_DEVICE_CONSTEXPR bool operator==(V value) const { return field == value.field && inner == value.inner; }

	HOST_DEVICE_CONSTEXPR bool operator!=(V value) const { return field != value.field || inner != value.inner; }

	HOST_DEVICE_CONSTEXPR V operator+() const { return V(+field, +inner); }

	HOST_DEVICE_CONSTEXPR V operator-() const { return V(-field, -inner); }

	HOST_DEVICE_CONSTEXPR V operator+(V value) const { return V(field + value.field, inner + value.inner); }

	HOST_DEVICE_CONSTEXPR V operator-(V value) const { return V(field - value.field, inner - value.inner); }

	HOST_DEVICE_CONSTEXPR V operator*(V value) const { return V(field * value.field, inner * value.inner); }

	HOST_DEVICE_CONSTEXPR V operator/(V value) const { return V(field / value.field, inner / value.inner); }

	HOST_DEVICE_CONSTEXPR V operator*(T value) const { return V(field * value, inner * value); }

	HOST_DEVICE_CONSTEXPR V operator/(T value) const { return V(field / value, inner / value); }

	HOST_DEVICE_CONSTEXPR V operator+=(V value) { return *this = *this + value; }

	HOST_DEVICE_CONSTEXPR V operator-=(V value) { return *this = *this - value; }

	HOST_DEVICE_CONSTEXPR V operator*=(V value) { return *this = *this * value; }

	HOST_DEVICE_CONSTEXPR V operator/=(V value) { return *this = *this / value; }

	HOST_DEVICE_CONSTEXPR V operator*=(T value) { return *this = *this * value; }

	HOST_DEVICE_CONSTEXPR V operator/=(T value) { return *this = *this / value; }

	friend constexpr std::ostream& operator<<(std::ostream& stream, V value) { return value.insert(stream << '(') << ')'; }

private:
	constexpr std::ostream& insert(std::ostream& stream) const { return inner.insert(stream << field << ", "); }

	T field;
	Vector<T, D - 1> inner;

	friend Vector<T, D + 1>;
};

template<typename T>
class Vector<T, 1>
{
private:
	using V = Vector<T, 1>;

public:
	HOST_DEVICE_CONSTEXPR Vector() : field{} {}

	HOST_DEVICE_CONSTEXPR Vector(const V& value) : field(value.field) {}

	HOST_DEVICE_CONSTEXPR explicit Vector(T value) : field(value) {}

	template<size_t I, std::enable_if_t<I == 0, bool> = true>
	HOST_DEVICE_CONSTEXPR T& at() { return field; }

	template<class U>
	HOST_DEVICE_CONSTEXPR Vector<U, 1> as() const { return Vector<U, 1>(static_cast<U>(field)); }

	HOST_DEVICE_CONSTEXPR T dot(V value) const { return field * value.field; }

	HOST_DEVICE_CONSTEXPR bool operator==(V value) const { return field == value.field; }

	HOST_DEVICE_CONSTEXPR bool operator!=(V value) const { return field != value.field; }

	HOST_DEVICE_CONSTEXPR V operator+() const { return V(+field); }

	HOST_DEVICE_CONSTEXPR V operator-() const { return V(-field); }

	HOST_DEVICE_CONSTEXPR V operator+(V value) const { return V(field + value.field); }

	HOST_DEVICE_CONSTEXPR V operator-(V value) const { return V(field - value.field); }

	HOST_DEVICE_CONSTEXPR V operator*(V value) const { return V(field * value.field); }

	HOST_DEVICE_CONSTEXPR V operator/(V value) const { return V(field / value.field); }

	HOST_DEVICE_CONSTEXPR V operator*(T value) const { return V(field * value); }

	HOST_DEVICE_CONSTEXPR V operator/(T value) const { return V(field / value); }

private:
	constexpr std::ostream& insert(std::ostream& stream) const { return stream << field; }

	T field;

	friend Vector<T, 2>;
};

#undef HOST_DEVICE_CONSTEXPR

} // cb
