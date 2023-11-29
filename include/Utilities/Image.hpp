#pragma once

#include "main.hpp"

namespace cb
{

class Image
{
public:
	static void write_png(const std::string& filename, uint32_t width, uint32_t height, const uint32_t* data);
};

} // cb
