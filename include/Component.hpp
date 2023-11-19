#pragma once

#include "main.hpp"
#include "Timer.hpp"

namespace cb
{

class Component
{
public:
	explicit Component(Application& application);

	virtual ~Component() = default;

	virtual void initialize() = 0;

	virtual void update(const Timer& timer) = 0;

protected:
	Application& application;
	sf::RenderWindow& window;
};

} // cb
