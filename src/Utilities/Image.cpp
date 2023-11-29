#include "Utilities/Image.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include <string>

namespace cb
{

void Image::write_png(const std::string& filename, uint32_t width, uint32_t height, const uint32_t* data)
{
	stbi_write_png(filename.data(), static_cast<int>(width), static_cast<int>(height), 4, data, static_cast<int>(sizeof(uint32_t) * width));
}

} // cb
