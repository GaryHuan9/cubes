#pragma once

#include "main.hpp"

namespace cb
{

class Component
{
public:
	explicit Component(Application& application) : application(application) {}

	virtual ~Component() = default;

	virtual void initialize() = 0;

	virtual void update() = 0;

protected:
	Application& application;
};

} // cb
