#pragma once

#include <cassert>
#include <cstdint>
#include <vector>
#include <string>
#include <memory>

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
class Engine;
class Ray;
class Accumulator;
class Path;
class TraceQuery;
class EscapedPacket;
class MaterialQuery;
class DiffuseParameters;
class ConductorParameters;
class DielectricParameters;
class EmissiveParameters;

class Controller;
class Scene;
class Camera;

template<typename T>
class CudaArray;

template<typename T>
class CudaVector;

template<typename T, size_t D>
class Vector;
typedef Vector<float, 2> Float2;
typedef Vector<float, 3> Float3;
typedef Vector<float, 4> Float4;
typedef Vector<int32_t, 2> Int2;
typedef Vector<int32_t, 3> Int3;
typedef Vector<int32_t, 4> Int4;
typedef Vector<uint32_t, 2> UInt2;
typedef Vector<uint32_t, 3> UInt3;
typedef Vector<uint32_t, 4> UInt4;

}
