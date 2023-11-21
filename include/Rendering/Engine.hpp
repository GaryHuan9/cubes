#pragma once

#include "main.hpp"
#include "Packets.hpp"
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

	void output(cudaSurfaceObject_t surface_object);

private:
	UInt2 resolution;
	CudaArray<Accumulator> accumulators;
	CudaVector<NewPathPackets> new_path_packets;
	CudaVector<TracePackets> trace_packets;
};

} // cb
