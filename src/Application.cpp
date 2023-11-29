#include "Application.hpp"
#include "Timer.hpp"
#include "Rendering/Renderer.hpp"
#include "Rendering/Engine.hpp"
#include "Scenic/Controller.hpp"

#include "SFML/System.hpp"
#include "SFML/Window.hpp"
#include "SFML/Graphics.hpp"
#include "imgui-SFML.h"
#include "imgui.h"

namespace cb
{

static void configure_spacing(ImGuiStyle& style);
static void configure_colors(ImGuiStyle& style);

Application::Application()
{
	window = std::make_unique<sf::RenderWindow>(sf::VideoMode{ 1920, 1080 }, "cubes");
	window->setFramerateLimit(240);
	ImGui::SFML::Init(*window, false);

	auto& style = ImGui::GetStyle();
	configure_spacing(style);
	configure_colors(style);

	auto& io = ImGui::GetIO();
	io.ConfigWindowsMoveFromTitleBarOnly = true;
	io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.Fonts->AddFontFromFileTTF("../ext/JetBrainsMono/JetBrainsMono-Bold.ttf", 16.0f);
	ImGui::SFML::UpdateFontTexture();

	make_component<Renderer>();
	make_component<Controller>();
}

Application::~Application()
{
	ImGui::SFML::Shutdown();
}

void Application::run()
{
	sf::Clock clock;
	Timer timer;

	for (auto& component : components) component->initialize();

	while (window->isOpen())
	{
		sf::Event event{};

		while (window->pollEvent(event))
		{
			ImGui::SFML::ProcessEvent(event);

			switch (event.type)
			{
				case sf::Event::Closed:
				{
					window->close();
					return;
				}
				case sf::Event::Resized:
				{
					sf::Vector2f size(sf::Vector2u(event.size.width, event.size.height));
					window->setView(sf::View(size / 2.0f, size));
					break;
				}
				case sf::Event::KeyPressed:
				{
					if (!window->hasFocus() || event.key.code != sf::Keyboard::Key::F1) break;
					find_component<Renderer>()->render_file("render.png");
					break;
				}
				default: break;
			}
		}

		sf::Time time = clock.restart();
		timer.update(time.asMicroseconds());
		ImGui::SFML::Update(*window, time);
		window->clear(sf::Color::Black);

		for (auto& component : components) component->update(timer);

		//		ImGui::ShowDemoWindow();
		ImGui::SFML::Render(*window);
		window->display();
	}
}

static void configure_spacing(ImGuiStyle& style)
{
	style.WindowPadding = ImVec2(8.0f, 8.0f);
	style.FramePadding = ImVec2(8.0f, 2.0f);
	style.CellPadding = ImVec2(4.0f, 2.0f);
	style.ItemSpacing = ImVec2(4.0f, 4.0f);
	style.ItemInnerSpacing = ImVec2(4.0f, 4.0f);
	style.TouchExtraPadding = ImVec2(0.0f, 0.0f);
	style.IndentSpacing = 20.0f;
	style.ScrollbarSize = 12.0f;
	style.GrabMinSize = 8.0f;

	style.WindowBorderSize = 1.0f;
	style.ChildBorderSize = 1.0f;
	style.PopupBorderSize = 1.0f;
	style.FrameBorderSize = 1.0f;
	style.TabBorderSize = 1.0f;

	style.WindowRounding = 1.0f;
	style.ChildRounding = 1.0f;
	style.FrameRounding = 1.0f;
	style.PopupRounding = 1.0f;
	style.ScrollbarRounding = 1.0f;
	style.GrabRounding = 1.0f;
	style.LogSliderDeadzone = 1.0f;
	style.TabRounding = 1.0f;

	style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
	style.WindowMenuButtonPosition = ImGuiDir_None;
	style.ColorButtonPosition = ImGuiDir_Left;
	style.ButtonTextAlign = ImVec2(0.5f, 0.5f);
	style.SelectableTextAlign = ImVec2(0.0f, 0.0f);
}

static void configure_colors(ImGuiStyle& style)
{
	constexpr float Alpha0 = 0.33f;
	constexpr float Alpha1 = 0.61f;
	const ImVec4 main = ImVec4(0.07450981f, 0.93333334f, 0.34117648, 1.0f);
	const ImVec4 white1 = ImVec4(0.98039216f, 0.9843137f, 1.0f, 1.0f);
	const ImVec4 white0 = ImVec4(0.627098f, 0.9644314f, 0.7430588, 1.0f);
	const ImVec4 background0 = ImVec4(0.078431375f, 0.08235294f, 0.09019608, 1.0f);
	const ImVec4 background1 = ImVec4(0.13725491f, 0.15294118f, 0.18039216, 1.0f);
	const ImVec4 contrast = ImVec4(0.21568628f, 0.23137255f, 0.24705882, 1.0f);

	auto with_alpha = [](const ImVec4& value, float alpha) { return ImVec4(value.x, value.y, value.z, alpha); };

	style.Colors[ImGuiCol_Text] = white1;
	style.Colors[ImGuiCol_TextDisabled] = white0;
	style.Colors[ImGuiCol_WindowBg] = background0;
	style.Colors[ImGuiCol_PopupBg] = background0;
	style.Colors[ImGuiCol_Border] = with_alpha(main, Alpha1);
	style.Colors[ImGuiCol_FrameBg] = ImVec4();
	style.Colors[ImGuiCol_FrameBgHovered] = contrast;
	style.Colors[ImGuiCol_FrameBgActive] = main;
	style.Colors[ImGuiCol_TitleBg] = background0;
	style.Colors[ImGuiCol_TitleBgActive] = contrast;
	style.Colors[ImGuiCol_TitleBgCollapsed] = background0;
	style.Colors[ImGuiCol_MenuBarBg] = background1;
	style.Colors[ImGuiCol_ScrollbarBg] = ImVec4();
	style.Colors[ImGuiCol_ScrollbarGrab] = background1;
	style.Colors[ImGuiCol_ScrollbarGrabHovered] = contrast;
	style.Colors[ImGuiCol_ScrollbarGrabActive] = main;
	style.Colors[ImGuiCol_CheckMark] = main;
	style.Colors[ImGuiCol_SliderGrab] = main;
	style.Colors[ImGuiCol_SliderGrabActive] = white0;
	style.Colors[ImGuiCol_Button] = ImVec4();
	style.Colors[ImGuiCol_ButtonHovered] = contrast;
	style.Colors[ImGuiCol_ButtonActive] = main;
	style.Colors[ImGuiCol_Header] = ImVec4();
	style.Colors[ImGuiCol_HeaderHovered] = contrast;
	style.Colors[ImGuiCol_HeaderActive] = main;
	style.Colors[ImGuiCol_Separator] = background1;
	style.Colors[ImGuiCol_SeparatorHovered] = contrast;
	style.Colors[ImGuiCol_SeparatorActive] = main;
	style.Colors[ImGuiCol_ResizeGrip] = ImVec4();
	style.Colors[ImGuiCol_ResizeGripHovered] = ImVec4();
	style.Colors[ImGuiCol_ResizeGripActive] = ImVec4();
	style.Colors[ImGuiCol_Tab] = background0;
	style.Colors[ImGuiCol_TabHovered] = main;
	style.Colors[ImGuiCol_TabActive] = main;
	style.Colors[ImGuiCol_TabUnfocused] = background0;
	style.Colors[ImGuiCol_TabUnfocusedActive] = contrast;
	style.Colors[ImGuiCol_DockingPreview] = contrast;
	style.Colors[ImGuiCol_DockingEmptyBg] = background0;
	style.Colors[ImGuiCol_PlotLines] = main;
	style.Colors[ImGuiCol_PlotLinesHovered] = white0;
	style.Colors[ImGuiCol_PlotHistogram] = main;
	style.Colors[ImGuiCol_PlotHistogramHovered] = white0;
	style.Colors[ImGuiCol_TableHeaderBg] = background1;
	style.Colors[ImGuiCol_TableBorderStrong] = with_alpha(main, Alpha1);
	style.Colors[ImGuiCol_TableBorderLight] = with_alpha(main, Alpha1);
	style.Colors[ImGuiCol_TableRowBgAlt] = with_alpha(background1, Alpha0);
	style.Colors[ImGuiCol_TextSelectedBg] = with_alpha(white1, Alpha0);
	style.Colors[ImGuiCol_DragDropTarget] = with_alpha(white1, Alpha1);
	style.Colors[ImGuiCol_NavHighlight] = with_alpha(white1, Alpha1);
	style.Colors[ImGuiCol_NavWindowingHighlight] = with_alpha(white1, Alpha1);
	style.Colors[ImGuiCol_NavWindowingDimBg] = with_alpha(white1, Alpha0);
	style.Colors[ImGuiCol_ModalWindowDimBg] = with_alpha(white1, Alpha0);
}

}
