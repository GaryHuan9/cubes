#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaArray.hpp"
#include "Utilities/CudaVector.hpp"

namespace cb::kernels
{

template<typename T>
using Array = cb::CudaArray<T>::Accessor;

template<typename T>
using List = cb::CudaVector<T>::Accessor;

__global__
void test(UInt2 resolution, Array<Accumulator> accumulators, float time);

__global__
void render_begin(UInt2 resolution, uint32_t offset, uint32_t length, Array<NewPathPackets> results);

__global__
void new_path(UInt2 resolution, Camera* camera, Array<NewPathPackets> packets, List<TracePackets> results);

__global__
void trace_rays(List<TracePackets> packets, Array<HitPacket> results);

__global__
void render_end(UInt2 resolution, List<TracePackets> packets, Array<HitPacket> results, Array<Accumulator> accumulators);

__global__
void output(UInt2 resolution, Array<Accumulator> accumulators, cudaSurfaceObject_t surface);

} // cb
