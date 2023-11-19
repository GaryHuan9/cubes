#pragma once

#include "main.hpp"

#include <typeinfo>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>
#include <ranges>

namespace cb
{

class Application
{
public:
	Application();
	~Application();

	void run();

	template<typename T>
	T* find_component() const
	{
		auto predicate = [](const auto& value) { return typeid(*value) == typeid(T); };
		auto iterator = std::ranges::find_if(components, predicate);
		return iterator == components.end() ? nullptr : static_cast<T*>(iterator->get());
	}

	template<typename T>
	T* make_component()
	{
		auto predicate = [](const auto& value) { return typeid(*value) == typeid(T); };
		size_t count = std::ranges::count_if(components, predicate);
		if (count != 0) throw std::invalid_argument("Already added component.");

		const auto& pointer = components.emplace_back(std::make_unique<T>(*this));
		return static_cast<T*>(pointer.get());
	}

	sf::RenderWindow* get_window()
	{
		return window.get();
	}

private:
	std::unique_ptr<sf::RenderWindow> window;
	std::vector<std::unique_ptr<Component>> components;
};

} // cb
