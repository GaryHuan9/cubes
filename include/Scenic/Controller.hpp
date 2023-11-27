#pragma once

#include "main.hpp"
#include "Component.hpp"
#include "Utilities/Vector.hpp"

namespace cb
{

class Controller : public Component
{
public:
	explicit Controller(Application& application);
	~Controller() override;

	void initialize() override;
	void update(const Timer& timer) override;

private:
	Int2 last_mouse_position;
	std::unique_ptr<Camera> camera;
	Engine* engine{};
};

} // cb
