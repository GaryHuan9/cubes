#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaArray.hpp"
#include "Utilities/CudaVector.hpp"

namespace cb
{

class Engine
{
public:
	Engine();
	~Engine();

	void change_resolution(const UInt2& new_resolution);

	void change_camera(const Camera& new_camera);

	void render();

	void output(cudaSurfaceObject_t surface_object) const;

private:
	static constexpr size_t Capacity = 1024 * 1024;

	UInt2 resolution;
	Camera* camera{};

	uint32_t scan_offset{};

	CudaArray<Accumulator> accumulators;
	CudaArray<NewPathPackets> new_path_packets;
	CudaVector<TracePackets> trace_packets;
	CudaArray<HitPacket> hit_packets;
};

} // cb
