#pragma once

#include <cassert>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

#define HOST_DEVICE_ENTRY __host__ __device__

namespace sf
{

class RenderWindow;
class Vertex;
class Text;
class Font;
class Sprite;
class Texture;

} // sf

namespace cb
{

class Application;
class Component;
class Renderer;

template<typename T, size_t D>
class Vector;
typedef Vector<float, 2> Float2;
typedef Vector<float, 3> Float3;
typedef Vector<float, 4> Float4;
typedef Vector<int32_t, 2> Int2;
typedef Vector<int32_t, 3> Int3;
typedef Vector<int32_t, 4> Int4;

} // cb
