#include "Component.hpp"
#include "Application.hpp"

namespace cb
{

Component::Component(Application& application) : application(application), window(*application.get_window()) {}

} // cb
