#pragma once

#include <memory>

namespace sf
{

class RenderWindow;

}

namespace cb
{

class Application
{
public:
	Application();
	~Application();

	void run();

private:
	std::unique_ptr<sf::RenderWindow> window;
};

} // cb
