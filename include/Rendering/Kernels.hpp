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
void new_path(Array<Path> paths, UInt2 resolution, uint32_t index_start, Camera* camera, List<TraceQuery> trace_queries);

__global__
void trace(List<TraceQuery> trace_queries);

__global__
void shade(List<TraceQuery> trace_queries, List<MaterialQuery> material_queries, List<EscapedPacket> escaped_packets);

__global__
void diffuse(List<MaterialQuery> material_queries);

__global__
void advance(List<MaterialQuery> material_queries, List<TraceQuery> trace_queries, Array<Path> paths, Array<Accumulator> accumulators);

__global__
void escaped(List<EscapedPacket> escaped_packets, Array<Path> paths, Array<Accumulator> accumulators);

__global__
void output(UInt2 resolution, Array<Accumulator> accumulators, cudaSurfaceObject_t surface);

}
