#pragma once

#include "main.hpp"
#include "Utilities/Vector.hpp"
#include "Utilities/CudaArray.hpp"
#include "Utilities/CudaVector.hpp"
#include "Utilities/CudaUtilities.hpp"

#include <curand_kernel.h>

namespace cb::kernels
{

template<typename T>
using Array = cb::CudaArray<T>::Accessor;

template<typename T>
using List = cb::CudaVector<T>::Accessor;

__global__
void new_random(Array<curandState> randoms);

template<typename... Ts>
__global__
void list_clear(List<Ts>... lists)
{
	([&] HOST_DEVICE { lists.clear(); }(), ...);
}

__global__
void new_path(Array<Path> paths, UInt2 resolution, uint32_t index_start, Array<curandState> randoms, Camera* camera, List<TraceQuery> trace_queries);

__global__
void trace(List<TraceQuery> trace_queries, List<MaterialQuery> material_queries, List<EscapedPacket> escaped_packets);

__global__
void pre_material(List<MaterialQuery> material_queries, Array<curandState> randoms);

__global__
void diffuse(List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, DiffuseParameters parameters);

__global__
void conductor(List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, ConductorParameters parameters);

__global__
void emissive(List<MaterialQuery> material_queries, Array<Path> paths, List<TraceQuery> trace_queries, EmissiveParameters parameters);

__global__
void escaped(List<EscapedPacket> escaped_packets, Array<Path> paths);

__global__
void accumulate(Array<Path> paths, uint32_t index_start, Array<Accumulator> accumulators);

__global__
void output_surface(UInt2 resolution, Array<Accumulator> accumulators, cudaSurfaceObject_t surface);

__global__
void output_buffer(UInt2 resolution, Array<Accumulator> accumulators, Array<uint32_t> array);

}
