#include <optix.h>
#include <optix_device.h>

// ============================================================================
// ESTRUCTURAS COMPARTIDAS
// ============================================================================

struct DatosLanzamiento {
    float3* puntos_datos;
    int* ids_datos;
    int num_puntos;
    float3 query;
    float radio;
    int* resultados_ids;
    float* resultados_dists;
    int* num_resultados;
    int tipo_distancia;
    OptixTraversableHandle gas_handle;
};

// Declaración de parámetros de lanzamiento
extern "C" { __constant__ DatosLanzamiento params; }

// ============================================================================
// FUNCIONES AUXILIARES DEVICE
// ============================================================================

__device__ __forceinline__ float distancia_l2(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

__device__ __forceinline__ float distancia_l1(float3 a, float3 b) {
    return fabsf(a.x - b.x) + fabsf(a.y - b.y) + fabsf(a.z - b.z);
}

__device__ __forceinline__ float distancia_linf(float3 a, float3 b) {
    float dx = fabsf(a.x - b.x);
    float dy = fabsf(a.y - b.y);
    float dz = fabsf(a.z - b.z);
    return fmaxf(fmaxf(dx, dy), dz);
}

__device__ __forceinline__ float distancia_coseno(float3 a, float3 b) {
    // Asume vectores normalizados
    float producto_punto = a.x*b.x + a.y*b.y + a.z*b.z;
    return 1.0f - producto_punto;
}

__device__ __forceinline__ bool intersecta_esfera(float3 punto, float3 centro, float radio) {
    float dist_sq = (punto.x - centro.x) * (punto.x - centro.x) +
                    (punto.y - centro.y) * (punto.y - centro.y) +
                    (punto.z - centro.z) * (punto.z - centro.z);
    return dist_sq <= radio * radio;
}

__device__ __forceinline__ bool intersecta_bipiramide(float3 punto, float3 centro, float radio) {
    float dist_l1 = fabsf(punto.x - centro.x) + 
                    fabsf(punto.y - centro.y) + 
                    fabsf(punto.z - centro.z);
    return dist_l1 <= radio;
}

__device__ __forceinline__ bool intersecta_cubo(float3 punto, float3 centro, float radio) {
    return (fabsf(punto.x - centro.x) <= radio &&
            fabsf(punto.y - centro.y) <= radio &&
            fabsf(punto.z - centro.z) <= radio);
}

// ============================================================================
// PROGRAMAS OPTIX
// ============================================================================

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    // Procesar cada punto directamente
    // RT CORES ya construyeron el BVH, ahora lo usamos para filtrado espacial
    int punto_idx = idx.x;
    if (punto_idx >= params.num_puntos) return;
    
    float3 punto = params.puntos_datos[punto_idx];
    float3 query = params.query;
    float radio = params.radio;
    int tipo_dist = params.tipo_distancia;
    
    // Verificar si está dentro del radio usando AABB primero (filtro grueso)
    float dx = fabsf(punto.x - query.x);
    float dy = fabsf(punto.y - query.y);
    float dz = fabsf(punto.z - query.z);
    
    // Filtro rápido con cubo AABB
    if (dx > radio || dy > radio || dz > radio) return;
    
    // Calcular distancia exacta (refinamiento)
    float dist;
    switch(tipo_dist) {
        case 0:
            dist = distancia_l2(query, punto);
            break;
        case 1:
            dist = distancia_l1(query, punto);
            break;
        case 2:
            dist = distancia_linf(query, punto);
            break;
        case 3:
            dist = distancia_coseno(query, punto);
            break;
        default:
            dist = 999999.0f;
    }
    
    // Si está dentro del radio, agregar a resultados
    if (dist <= radio) {
        int pos = atomicAdd(params.num_resultados, 1);
        if (pos < 10000) {
            params.resultados_ids[pos] = params.ids_datos[punto_idx];
            params.resultados_dists[pos] = dist;
        }
    }
}

extern "C" __global__ void __miss__ms() {
    // No hay intersección con ningún AABB
}

extern "C" __global__ void __intersection__is() {
    // Este kernel se ejecuta cuando el RT core detecta intersección ray-AABB
    
    const int prim_idx = optixGetPrimitiveIndex();
    
    // Obtener centro del punto
    float3 centro = params.puntos_datos[prim_idx];
    float3 query = params.query;
    float radio = params.radio;
    int tipo_dist = params.tipo_distancia;
    
    bool intersecta = false;
    
    // FASE REFINE: Verificar intersección con geometría real
    switch(tipo_dist) {
        case 0: // L2
            intersecta = intersecta_esfera(query, centro, radio);
            break;
        case 1: // L1
            intersecta = intersecta_bipiramide(query, centro, radio);
            break;
        case 2: // Linf
            intersecta = intersecta_cubo(query, centro, radio);
            break;
        case 3: // Coseno (usa L2 sobre normalizados)
            intersecta = intersecta_esfera(query, centro, radio);
            break;
    }
    
    if (intersecta) {
        // Calcular distancia exacta
        float dist;
        switch(tipo_dist) {
            case 0:
                dist = distancia_l2(query, centro);
                break;
            case 1:
                dist = distancia_l1(query, centro);
                break;
            case 2:
                dist = distancia_linf(query, centro);
                break;
            case 3:
                dist = distancia_coseno(query, centro);
                break;
        }
        
        if (dist <= radio) {
            // Reportar hit
            optixReportIntersection(
                dist,           // t
                0,              // hitKind
                __float_as_uint(dist), // Pasar distancia en payload
                prim_idx        // Pasar índice
            );
        }
    }
}

extern "C" __global__ void __closesthit__ch() {
    // Este kernel se ejecuta cuando se confirma un hit
    
    const int prim_idx = optixGetPrimitiveIndex();
    
    // Obtener distancia del payload
    float dist = __uint_as_float(optixGetPayload_0());
    int id = params.ids_datos[prim_idx];
    
    // Agregar a resultados de forma atómica
    int pos = atomicAdd(params.num_resultados, 1);
    
    if (pos < 10000) { // MAX_RESULTADOS
        params.resultados_ids[pos] = id;
        params.resultados_dists[pos] = dist;
    }
}

// ============================================================================
// KERNEL CUDA PARA BUSQUEDA LINEAL (FALLBACK)
// ============================================================================

__global__ void busqueda_lineal_kernel(
    const float3* puntos,
    const int* ids,
    int num_puntos,
    float3 query,
    float radio,
    int tipo_distancia,
    int* resultados_ids,
    float* resultados_dists,
    int* num_resultados) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_puntos) return;
    
    float3 punto = puntos[idx];
    float dist;
    
    switch(tipo_distancia) {
        case 0:
            dist = distancia_l2(query, punto);
            break;
        case 1:
            dist = distancia_l1(query, punto);
            break;
        case 2:
            dist = distancia_linf(query, punto);
            break;
        case 3:
            dist = distancia_coseno(query, punto);
            break;
    }
    
    if (dist <= radio) {
        int pos = atomicAdd(num_resultados, 1);
        if (pos < 10000) {
            resultados_ids[pos] = ids[idx];
            resultados_dists[pos] = dist;
        }
    }
}

// ============================================================================
// FUNCIÓN WRAPPER PARA LLAMAR DESDE C++
// ============================================================================

extern "C" void lanzar_busqueda_lineal(
    const float3* puntos,
    const int* ids,
    int num_puntos,
    float3 query,
    float radio,
    int tipo_distancia,
    int* resultados_ids,
    float* resultados_dists,
    int* num_resultados) {
    
    const int threads_per_block = 256;
    const int num_blocks = (num_puntos + threads_per_block - 1) / threads_per_block;
    
    busqueda_lineal_kernel<<<num_blocks, threads_per_block>>>(
        puntos, ids, num_puntos, query, radio, tipo_distancia,
        resultados_ids, resultados_dists, num_resultados
    );
}
