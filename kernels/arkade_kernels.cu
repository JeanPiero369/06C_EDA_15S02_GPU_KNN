/**
 * ============================================================================
 * ARKADE KERNELS - Implementacion OptiX con RT Cores
 * ============================================================================
 * 
 * Implementacion basada en el paper:
 * "Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing"
 * 
 * METODOLOGIA FILTER-REFINE:
 * 
 * FASE FILTER (RT Cores - Hardware):
 *   - Construir GAS (Geometry Acceleration Structure) = BVH sobre AABBs
 *   - Los AABBs encierran las geometrias de cada metrica
 *   - RT Cores aceleran la construccion y traversia del BVH
 * 
 * FASE REFINE (Shader Cores - Software):
 *   - Para cada candidato dentro del AABB, verificar geometria real
 *   - Calcular distancia exacta segun metrica
 *   - Filtrar falsos positivos del AABB
 * 
 * GEOMETRIAS POR METRICA:
 *   - L2 (Euclidiana): Esfera de radio r (~52% ocupacion AABB)
 *   - L1 (Manhattan): Octaedro/Bipiramide de radio r (~33% ocupacion AABB)
 *   - Linf (Chebyshev): Cubo de radio r (100% ocupacion AABB - sin fase refine)
 *   - Coseno: Esfera en espacio normalizado
 * 
 * ============================================================================
 */

#include <optix.h>
#include <optix_device.h>

// ============================================================================
// ESTRUCTURAS COMPARTIDAS CPU-GPU
// ============================================================================

struct DatosLanzamiento {
    // Datos del dataset
    float3* puntos_datos;           // Coordenadas de puntos en GPU
    int* ids_datos;                 // IDs originales de puntos
    int num_puntos;                 // Cantidad de puntos
    
    // BATCH de queries (para paralelismo)
    float3* queries;                // Array de queries
    int num_queries;                // Numero de queries en el batch
    float radio;                    // Radio de busqueda
    
    // Buffers de resultados (uno por query)
    // Layout: [query0_result0, query0_result1, ..., query1_result0, ...]
    int* resultados_ids;            // IDs de vecinos: [num_queries * k]
    float* resultados_dists;        // Distancias: [num_queries * k]
    int* num_resultados;            // Contador por query: [num_queries]
    int k;                          // Numero de vecinos por query
    
    // Configuracion
    int tipo_distancia;             // 0=L2, 1=L1, 2=Linf, 3=Coseno
    OptixTraversableHandle gas_handle;
};

// Parametros de lanzamiento en memoria constante
extern "C" { __constant__ DatosLanzamiento params; }

// ============================================================================
// MAX-HEAP PARA K VECINOS MAS CERCANOS
// ============================================================================
// Usamos un max-heap para mantener los K vecinos mas cercanos.
// El elemento en la raiz (indice 0) es el mas lejano de los K.
// Cuando llega un nuevo candidato:
//   - Si heap no esta lleno: insertar
//   - Si candidato < raiz: reemplazar raiz y heapify-down
//   - Si candidato >= raiz: ignorar (no es top-K)

__device__ __forceinline__ void heap_sift_down(
    float* dists, int* ids, int n, int i
) {
    while (true) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;
        
        if (left < n && dists[left] > dists[largest]) {
            largest = left;
        }
        if (right < n && dists[right] > dists[largest]) {
            largest = right;
        }
        
        if (largest == i) break;
        
        // Swap
        float tmp_d = dists[i];
        int tmp_id = ids[i];
        dists[i] = dists[largest];
        ids[i] = ids[largest];
        dists[largest] = tmp_d;
        ids[largest] = tmp_id;
        
        i = largest;
    }
}

__device__ __forceinline__ void heap_sift_up(
    float* dists, int* ids, int i
) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (dists[i] <= dists[parent]) break;
        
        // Swap
        float tmp_d = dists[i];
        int tmp_id = ids[i];
        dists[i] = dists[parent];
        ids[i] = ids[parent];
        dists[parent] = tmp_d;
        ids[parent] = tmp_id;
        
        i = parent;
    }
}

// Insertar en max-heap, manteniendo solo los K mas cercanos
// Ahora cada query tiene su propio heap (query_idx)
__device__ void heap_insert(int query_idx, float dist, int id) {
    int k = params.k;
    int offset = query_idx * k;
    float* dists = params.resultados_dists + offset;
    int* ids = params.resultados_ids + offset;
    int* count = params.num_resultados + query_idx;
    
    // Operacion atomica para obtener posicion actual
    int current_count = atomicAdd(count, 0);  // Leer sin modificar
    
    if (current_count < k) {
        // Heap no esta lleno, insertar directamente
        int pos = atomicAdd(count, 1);
        if (pos < k) {
            dists[pos] = dist;
            ids[pos] = id;
            // Sift up para mantener propiedad de max-heap
            heap_sift_up(dists, ids, pos);
        }
    } else {
        // Heap lleno, comparar con raiz (maximo actual)
        if (dist < dists[0]) {
            // Nuevo candidato es mejor, reemplazar raiz
            dists[0] = dist;
            ids[0] = id;
            // Sift down para restaurar propiedad de max-heap
            heap_sift_down(dists, ids, k, 0);
        }
    }
}

// ============================================================================
// FUNCIONES DE DISTANCIA (FASE REFINE)
// ============================================================================

/**
 * Distancia Euclidiana (L2)
 * d(a,b) = sqrt((ax-bx)^2 + (ay-by)^2 + (az-bz)^2)
 */
__device__ __forceinline__ float distancia_l2(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

/**
 * Distancia Manhattan (L1)
 * d(a,b) = |ax-bx| + |ay-by| + |az-bz|
 */
__device__ __forceinline__ float distancia_l1(float3 a, float3 b) {
    return fabsf(a.x - b.x) + fabsf(a.y - b.y) + fabsf(a.z - b.z);
}

/**
 * Distancia Chebyshev (Linf)
 * d(a,b) = max(|ax-bx|, |ay-by|, |az-bz|)
 */
__device__ __forceinline__ float distancia_linf(float3 a, float3 b) {
    float dx = fabsf(a.x - b.x);
    float dy = fabsf(a.y - b.y);
    float dz = fabsf(a.z - b.z);
    return fmaxf(fmaxf(dx, dy), dz);
}

/**
 * Distancia Angular (Coseno)
 * d(a,b) = arccos(a . b) donde a,b estan normalizados
 * Rango: [0, pi]
 */
__device__ __forceinline__ float distancia_coseno(float3 a, float3 b) {
    // Los vectores ya vienen normalizados desde el host
    float producto_punto = a.x*b.x + a.y*b.y + a.z*b.z;
    // Clamp para evitar errores numericos con arccos
    producto_punto = fminf(fmaxf(producto_punto, -1.0f), 1.0f);
    return acosf(producto_punto);
}

// ============================================================================
// FUNCIONES DE INTERSECCION GEOMETRICA (FASE REFINE)
// ============================================================================

/**
 * Test de interseccion punto-esfera (L2)
 * Verifica si el punto query esta dentro de la esfera de radio r centrada en centro
 * Ocupacion AABB: ~52% (esfera inscrita en cubo)
 */
__device__ __forceinline__ bool intersecta_esfera(float3 query, float3 centro, float radio) {
    float dx = query.x - centro.x;
    float dy = query.y - centro.y;
    float dz = query.z - centro.z;
    float dist_sq = dx*dx + dy*dy + dz*dz;
    return dist_sq <= radio * radio;
}

/**
 * Test de interseccion punto-octaedro/bipiramide (L1)
 * Verifica si el punto query esta dentro del octaedro de radio r centrado en centro
 * Geometria: |x-cx| + |y-cy| + |z-cz| <= r
 * Ocupacion AABB: ~33% (octaedro inscrito en cubo)
 */
__device__ __forceinline__ bool intersecta_octaedro(float3 query, float3 centro, float radio) {
    float dist_l1 = fabsf(query.x - centro.x) + 
                    fabsf(query.y - centro.y) + 
                    fabsf(query.z - centro.z);
    return dist_l1 <= radio;
}

/**
 * Test de interseccion punto-cubo (Linf)
 * Verifica si el punto query esta dentro del cubo de lado 2r centrado en centro
 * Geometria: max(|x-cx|, |y-cy|, |z-cz|) <= r
 * Ocupacion AABB: 100% (cubo = AABB, caso optimo sin falsos positivos)
 */
__device__ __forceinline__ bool intersecta_cubo(float3 query, float3 centro, float radio) {
    return (fabsf(query.x - centro.x) <= radio &&
            fabsf(query.y - centro.y) <= radio &&
            fabsf(query.z - centro.z) <= radio);
}

// ============================================================================
// PROGRAMA RAYGEN - Punto de entrada principal (PARALELIZADO)
// ============================================================================
/**
 * __raygen__rg
 * 
 * METODOLOGIA ARKADE - RADIUS QUERY CON RT CORES (ESTRICTA)
 * 
 * EQUIVALENCIA MATEMATICA:
 *   Buscar puntos del dataset dentro de radio R desde query
 *   ≡ Query (punto) intersecta AABBs de radio R centrados en cada punto del dataset
 * 
 * IMPLEMENTACION:
 *   - AABBs: Tamaño = radio, centrados en cada punto del dataset
 *   - Rayo: PUNTUAL (longitud casi cero) desde la query
 *   - RT Cores: Retornan todos los AABBs que el rayo puntual intersecta = CANDIDATOS
 *   - Refine: Verificar geometria exacta en __intersection__is
 * 
 * PARALELIZACION:
 *   - Lanzamos num_queries threads en paralelo
 *   - Cada thread procesa UNA query usando optixGetLaunchIndex()
 *   - Los resultados se guardan en buffers separados por query
 */
extern "C" __global__ void __raygen__rg() {
    
    // Obtener indice de esta query del launch (PARALELIZACION!)
    const uint3 idx = optixGetLaunchIndex();
    const int query_idx = idx.x;
    
    // Verificar limites
    if (query_idx >= params.num_queries) return;
    
    // Obtener la query de este thread
    float3 query = params.queries[query_idx];
    float radio = params.radio;
    
    // ========================================================================
    // RAYO PUNTUAL DESDE LA QUERY
    // ========================================================================
    // - Origen: query point
    // - Direccion: cualquiera (usamos +X normalizado)
    // - t_min = 0, t_max = epsilon (rayo de longitud casi 0)
    //
    // Este rayo "toca" todos los AABBs que CONTIENEN el punto query.
    // OptiX llamara a __intersection__is para cada AABB candidato.
    
    float3 ray_origin = query;
    float3 ray_direction = make_float3(1.0f, 0.0f, 0.0f);  // Direccion normalizada
    
    // Payload: pasar el query_idx al intersection program
    unsigned int p0 = query_idx;
    unsigned int p1 = 0;
    
    // ========================================================================
    // optixTrace - RT Cores detectan AABBs que contienen la query
    // ========================================================================
    optixTrace(
        params.gas_handle,          // Handle del GAS (BVH construido sobre AABBs)
        ray_origin,                 // Origen = query point
        ray_direction,              // Direccion (arbitraria)
        0.0f,                       // t_min = 0
        1e-6f,                      // t_max = epsilon (rayo PUNTUAL)
        0.0f,                       // rayTime
        OptixVisibilityMask(255),   // Visibilidad
        OPTIX_RAY_FLAG_NONE,        // Flags
        0, 1, 0,                    // SBT offset, stride, miss index
        p0, p1                      // Payload con query_idx
    );
}

// ============================================================================
// PROGRAMA MISS - No hay interseccion
// ============================================================================
/**
 * __miss__ms
 * Se ejecuta cuando un rayo no intersecta ningun AABB.
 */
extern "C" __global__ void __miss__ms() {
    // Sin operacion
}

// ============================================================================
// PROGRAMA INTERSECTION - FASE REFINE con geometria exacta
// ============================================================================
/**
 * __intersection__is
 * 
 * Este programa se llama cuando RT Cores detectan que el rayo (puntual)
 * esta dentro de un AABB candidato.
 * 
 * FASE REFINE:
 *   - Verificar si query esta dentro de la GEOMETRIA REAL (no solo AABB)
 *   - Para L2: esfera (52% ocupacion AABB)
 *   - Para L1: octaedro (33% ocupacion AABB)
 *   - Para Linf: cubo (100% ocupacion AABB = sin falsos positivos)
 * 
 * Si pasa el test, reportamos interseccion con t_hit = 0 (rayo puntual).
 */
extern "C" __global__ void __intersection__is() {
    // Obtener indice de la primitiva (punto del dataset)
    const int prim_idx = optixGetPrimitiveIndex();
    
    // Obtener query_idx del payload
    const int query_idx = optixGetPayload_0();
    
    // Obtener punto del dataset y query de este thread
    float3 punto = params.puntos_datos[prim_idx];
    float3 query = params.queries[query_idx];
    float radio = params.radio;
    int tipo_dist = params.tipo_distancia;
    
    // ========================================================================
    // FASE REFINE: Verificar geometria exacta segun metrica
    // ========================================================================
    
    float dist = 0.0f;
    bool dentro = false;
    
    switch(tipo_dist) {
        case 0: // L2 - Esfera
            dist = distancia_l2(query, punto);
            dentro = (dist <= radio);
            break;
            
        case 1: // L1 - Octaedro
            dist = distancia_l1(query, punto);
            dentro = (dist <= radio);
            break;
            
        case 2: // Linf - Cubo (AABB = geometria, sin falsos positivos)
            dist = distancia_linf(query, punto);
            dentro = (dist <= radio);
            break;
            
        case 3: // Coseno
            dist = distancia_coseno(query, punto);
            dentro = (dist <= radio);
            break;
    }
    
    // Reportar interseccion si pasa el test de geometria
    if (dentro) {
        // Para rayo puntual, usamos t_hit = 0 (origen del rayo)
        // Atributos: distancia real y indice del punto
        optixReportIntersection(
            0.0f,                               // t_hit = 0 (rayo puntual)
            0,                                  // hitKind
            __float_as_uint(dist),              // attr0: distancia
            static_cast<unsigned int>(prim_idx) // attr1: indice
        );
    }
}

// ============================================================================
// PROGRAMA ANYHIT - Capturar K vecinos mas cercanos (radius query + heap)
// ============================================================================
/**
 * __anyhit__ah
 * 
 * Para k-NN necesitamos los K puntos mas cercanos dentro del radio.
 * Usamos un MAX-HEAP para mantener solo los K mejores candidatos.
 * 
 * Este programa se ejecuta para cada interseccion confirmada.
 * Insertamos en el heap y llamamos optixIgnoreIntersection() para
 * continuar buscando mas hits.
 */
extern "C" __global__ void __anyhit__ah() {
    // Obtener query_idx del payload
    const int query_idx = optixGetPayload_0();
    
    // Obtener atributos desde __intersection__is
    float dist = __uint_as_float(optixGetAttribute_0());
    int prim_idx = static_cast<int>(optixGetAttribute_1());
    
    // Obtener ID del punto
    int id = params.ids_datos[prim_idx];
    
    // Insertar en max-heap de esta query (mantiene solo los K mas cercanos)
    heap_insert(query_idx, dist, id);
    
    // Continuar buscando mas hits
    optixIgnoreIntersection();
}

// ============================================================================
// PROGRAMA CLOSESTHIT - Fallback (no usado con anyhit)
// ============================================================================
/**
 * __closesthit__ch
 * 
 * Fallback si anyhit no esta configurado.
 * Usa el mismo mecanismo de heap.
 */
extern "C" __global__ void __closesthit__ch() {
    // Obtener query_idx del payload
    const int query_idx = optixGetPayload_0();
    
    // Obtener atributos
    float dist = __uint_as_float(optixGetAttribute_0());
    int prim_idx = static_cast<int>(optixGetAttribute_1());
    int id = params.ids_datos[prim_idx];
    
    // Insertar en max-heap de esta query
    heap_insert(query_idx, dist, id);
}
