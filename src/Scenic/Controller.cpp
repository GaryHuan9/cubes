#include "Scenic/Controller.hpp"
#include "Scenic/Camera.hpp"
#include "Rendering/Renderer.hpp"
#include "Rendering/Engine.hpp"
#include "Application.hpp"

#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"

namespace cb
{

Controller::Controller(Application& application) : Component(application),
                                                   camera(std::make_unique<Camera>()) {}

Controller::~Controller() = default;

void Controller::initialize()
{
	engine = application.find_component<Renderer>()->get_engine();

	camera->set_position(Float3(25.0f, 35.0f, -50.0f));
	camera->set_rotation(Float2(27.0f, -27.0f));
	camera->set_field_of_view(75.0f);
	engine->change_camera(*camera);
}

void Controller::update(const Timer& timer)
{
	if (!window.hasFocus()) return;

	Int3 position_delta;
	Int2 rotation_delta;

	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W)) position_delta += Int3(0, 0, 1);
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S)) position_delta += Int3(0, 0, -1);
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A)) position_delta += Int3(-1, 0, 0);
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D)) position_delta += Int3(1, 0, 0);
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space)) position_delta += Int3(0, 1, 0);
	if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::LShift)) position_delta += Int3(0, -1, 0);

	if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left))
	{
		auto position = sf::Mouse::getPosition();
		Int2 mouse_position(position.x, position.y);

		if (last_mouse_position != Int2())
		{
			Int2 delta = mouse_position - last_mouse_position;
			rotation_delta = Int2(delta.y(), delta.x());
		}

		last_mouse_position = mouse_position;
	}
	else last_mouse_position = Int2();

	if (position_delta == Int3() && rotation_delta == Int2()) return;

	camera->move_position(Float3(position_delta) * 3.5f * Timer::as_float(timer.frame_time()));
	camera->set_rotation(camera->get_rotation() + Float2(rotation_delta) * 0.1f);
	engine->change_camera(*camera);
}

}
