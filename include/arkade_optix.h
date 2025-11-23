#ifndef ARKADE_OPTIX_H
#define ARKADE_OPTIX_H

#define _USE_MATH_DEFINES  // Para M_PI en Windows
#include <cmath>
#include "utilidades.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iterator>

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

struct AABBData {
    float3 centro;
    float radio;
    int id_punto;
};

// ============================================================================
// CLASE ARKADE CON OPTIX
// ============================================================================

class ArkadeOptiX {
public:
    // Constantes para tipos de distancia
    static const int DIST_L2 = 0;
    static const int DIST_L1 = 1;
    static const int DIST_LINF = 2;
    static const int DIST_COSINE = 3;

private:
    OptixDeviceContext contexto_optix;
    OptixModule modulo_optix;
    OptixProgramGroup programa_raygen;
    OptixProgramGroup programa_miss;
    OptixProgramGroup programa_hitgroup;
    OptixPipeline pipeline;
    OptixShaderBindingTable sbt;
    
    CUdeviceptr buffer_gas; // Geometry Acceleration Structure
    OptixTraversableHandle gas_handle; // Handle para ray tracing
    void* buffer_lanzamiento;
    void* buffer_puntos;
    void* buffer_ids;
    void* buffer_resultados_ids;
    void* buffer_resultados_dists;
    void* buffer_num_resultados;
    
    CUdeviceptr raygen_record;
    CUdeviceptr miss_record;
    CUdeviceptr hitgroup_record;
    
    // Buffers persistentes para resultados (evitar malloc/free por query)
    int* d_resultados_ids_persistente;
    float* d_resultados_dists_persistente;
    int* d_num_resultados_persistente;
    float3* d_queries_persistente;  // Buffer de queries en GPU
    CUdeviceptr d_params_persistente;
    bool buffers_inicializados;
    bool gas_construido;
    float radio_gas;  // Radio con el que se construyo el GAS
    int max_batch_size;  // Tamaño máximo del batch
    int k_actual;  // K actual para el que se asignaron los buffers
    
    std::vector<Punto3D> datos;
    std::vector<Punto3D> datos_normalizados;
    int tipo_distancia;
    
    void verificar_cuda(cudaError_t resultado, const char* mensaje) {
        if (resultado != cudaSuccess) {
            throw std::runtime_error(std::string(mensaje) + ": " + cudaGetErrorString(resultado));
        }
    }
    
    void verificar_cu(CUresult resultado, const char* mensaje) {
        if (resultado != CUDA_SUCCESS) {
            const char* nombre_error;
            cuGetErrorName(resultado, &nombre_error);
            throw std::runtime_error(std::string(mensaje) + ": " + nombre_error);
        }
    }
    
    void verificar_optix(OptixResult resultado, const char* mensaje) {
        if (resultado != OPTIX_SUCCESS) {
            throw std::runtime_error(std::string(mensaje) + ": " + optixGetErrorString(resultado));
        }
    }
    
    static void callback_log_optix(unsigned int nivel, const char* etiqueta, 
                                    const char* mensaje, void* /*datos*/) {
        std::cerr << "[" << nivel << "][" << etiqueta << "]: " << mensaje << std::endl;
    }
    
public:
    ArkadeOptiX(int tipo_dist) : tipo_distancia(tipo_dist),
        contexto_optix(nullptr), modulo_optix(nullptr),
        programa_raygen(nullptr), programa_miss(nullptr), programa_hitgroup(nullptr),
        pipeline(nullptr), buffer_gas(0), gas_handle(0),
        buffer_lanzamiento(nullptr), buffer_puntos(nullptr), buffer_ids(nullptr),
        buffer_resultados_ids(nullptr), buffer_resultados_dists(nullptr),
        buffer_num_resultados(nullptr),
        raygen_record(0), miss_record(0), hitgroup_record(0),
        d_resultados_ids_persistente(nullptr), d_resultados_dists_persistente(nullptr),
        d_num_resultados_persistente(nullptr), d_queries_persistente(nullptr),
        d_params_persistente(0),
        buffers_inicializados(false), gas_construido(false), radio_gas(0.0f),
        max_batch_size(0), k_actual(0) {
        sbt = {};
        inicializar_optix();
    }
    
    ~ArkadeOptiX() {
        limpiar_recursos();
    }
    
    void inicializar_optix();
    void cargar_datos(const std::vector<Punto3D>& puntos);
    void construir_gas();
    void crear_pipeline();
    void configurar_sbt();
    void construir_gas_con_radio(float radio); // Construir GAS una vez con radio específico
    std::vector<ResultadoVecino> buscar_radius(const Punto3D& query, float radio);
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, int k, float radio);
    void limpiar_recursos();
};

// ============================================================================
// IMPLEMENTACION
// ============================================================================

void ArkadeOptiX::inicializar_optix() {
    // Inicializar CUDA Driver API
    verificar_cu(cuInit(0), "Inicializar CUDA Driver API");
    
    // Inicializar CUDA Runtime
    verificar_cuda(cudaFree(0), "Inicializar CUDA Runtime");
    
    // Inicializar OptiX
    verificar_optix(optixInit(), "Inicializar OptiX");
    
    // Crear contexto OptiX
    OptixDeviceContextOptions opciones = {};
    opciones.logCallbackFunction = &callback_log_optix;
    opciones.logCallbackLevel = 4;
    
    CUcontext contexto_cuda = 0; // Se usa el contexto actual
    verificar_optix(
        optixDeviceContextCreate(contexto_cuda, &opciones, &contexto_optix),
        "Crear contexto OptiX"
    );
    
    std::cout << "OptiX inicializado correctamente" << std::endl;
}

void ArkadeOptiX::cargar_datos(const std::vector<Punto3D>& puntos) {
    datos = puntos;
    
    // Para distancia coseno, normalizar
    if (tipo_distancia == 3) {
        datos_normalizados.reserve(datos.size());
        for (const auto& p : datos) {
            datos_normalizados.push_back(p.normalizar());
        }
    }
    
    // Preparar datos para GPU
    const std::vector<Punto3D>& datos_usar = 
        (tipo_distancia == 3) ? datos_normalizados : datos;
    
    size_t tam_puntos = datos_usar.size() * sizeof(float3);
    size_t tam_ids = datos_usar.size() * sizeof(int);
    
    std::vector<float3> puntos_gpu(datos_usar.size());
    std::vector<int> ids_gpu(datos_usar.size());
    
    for (size_t i = 0; i < datos_usar.size(); i++) {
        puntos_gpu[i] = make_float3(datos_usar[i].x, datos_usar[i].y, datos_usar[i].z);
        ids_gpu[i] = datos[i].id;
    }
    
    // Copiar a GPU
    verificar_cuda(cudaMalloc(&buffer_puntos, tam_puntos), "Asignar buffer_puntos");
    verificar_cuda(cudaMalloc(&buffer_ids, tam_ids), "Asignar buffer_ids");
    
    verificar_cuda(
        cudaMemcpy(buffer_puntos, puntos_gpu.data(), 
                   tam_puntos, cudaMemcpyHostToDevice),
        "Copiar puntos a GPU"
    );
    
    verificar_cuda(
        cudaMemcpy(buffer_ids, ids_gpu.data(), 
                   tam_ids, cudaMemcpyHostToDevice),
        "Copiar IDs a GPU"
    );
    
    std::cout << "Datos cargados en GPU: " << datos.size() << " puntos" << std::endl;
}

void ArkadeOptiX::construir_gas() {
    // Construir GAS con radio inicial grande que se reutilizará
    std::cout << "Inicializando pipeline OptiX (GAS se construirá en batch)" << std::endl;
    
    crear_pipeline();
    configurar_sbt();
}

void ArkadeOptiX::crear_pipeline() {
    // Cargar módulo PTX
    OptixModuleCompileOptions opciones_modulo = {};
    opciones_modulo.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    opciones_modulo.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    opciones_modulo.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    
    OptixPipelineCompileOptions opciones_pipeline = {};
    opciones_pipeline.usesMotionBlur = false;
    opciones_pipeline.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    opciones_pipeline.numPayloadValues = 4;
    opciones_pipeline.numAttributeValues = 2;
    opciones_pipeline.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    opciones_pipeline.pipelineLaunchParamsVariableName = "params";
    
    // Leer PTX desde archivo (primero intentar directorio actual, luego build/bin)
    std::ifstream archivo_ptx("arkade_kernels.ptx", std::ios::binary);
    if (!archivo_ptx.is_open()) {
        archivo_ptx.open("build/bin/arkade_kernels.ptx", std::ios::binary);
    }
    if (!archivo_ptx.is_open()) {
        throw std::runtime_error("No se pudo abrir arkade_kernels.ptx (buscar en . o build/bin/)");
    }
    
    std::string ptx_code((std::istreambuf_iterator<char>(archivo_ptx)),
                         std::istreambuf_iterator<char>());
    archivo_ptx.close();
    
    char log[2048];
    size_t sizeof_log = sizeof(log);
    
    verificar_optix(
        optixModuleCreate(
            contexto_optix,
            &opciones_modulo,
            &opciones_pipeline,
            ptx_code.c_str(),
            ptx_code.size(),
            log, &sizeof_log,
            &modulo_optix
        ),
        "Crear módulo PTX"
    );
    
    // Crear program groups
    OptixProgramGroupOptions opciones_pg = {};
    
    // Raygen program
    OptixProgramGroupDesc desc_raygen = {};
    desc_raygen.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc_raygen.raygen.module = modulo_optix;
    desc_raygen.raygen.entryFunctionName = "__raygen__rg";
    
    verificar_optix(
        optixProgramGroupCreate(contexto_optix, &desc_raygen, 1,
                                &opciones_pg, log, &sizeof_log,
                                &programa_raygen),
        "Crear programa raygen"
    );
    
    // Miss program
    OptixProgramGroupDesc desc_miss = {};
    desc_miss.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc_miss.miss.module = modulo_optix;
    desc_miss.miss.entryFunctionName = "__miss__ms";
    
    verificar_optix(
        optixProgramGroupCreate(contexto_optix, &desc_miss, 1,
                                &opciones_pg, log, &sizeof_log,
                                &programa_miss),
        "Crear programa miss"
    );
    
    // Hitgroup program con ANYHIT para capturar TODOS los hits (radius query)
    OptixProgramGroupDesc desc_hitgroup = {};
    desc_hitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc_hitgroup.hitgroup.moduleCH = modulo_optix;
    desc_hitgroup.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    desc_hitgroup.hitgroup.moduleAH = modulo_optix;  // ANYHIT program
    desc_hitgroup.hitgroup.entryFunctionNameAH = "__anyhit__ah";
    desc_hitgroup.hitgroup.moduleIS = modulo_optix;
    desc_hitgroup.hitgroup.entryFunctionNameIS = "__intersection__is";
    
    verificar_optix(
        optixProgramGroupCreate(contexto_optix, &desc_hitgroup, 1,
                                &opciones_pg, log, &sizeof_log,
                                &programa_hitgroup),
        "Crear programa hitgroup"
    );
    
    // Link pipeline
    OptixProgramGroup grupos_programa[] = {
        programa_raygen,
        programa_miss,
        programa_hitgroup
    };
    
    OptixPipelineLinkOptions opciones_link = {};
    opciones_link.maxTraceDepth = 1;
    
    verificar_optix(
        optixPipelineCreate(
            contexto_optix,
            &opciones_pipeline,
            &opciones_link,
            grupos_programa,
            sizeof(grupos_programa) / sizeof(grupos_programa[0]),
            log, &sizeof_log,
            &pipeline
        ),
        "Crear pipeline"
    );
    
    std::cout << "Pipeline OptiX creado con RT Cores" << std::endl;
}

void ArkadeOptiX::configurar_sbt() {
    // SBT records solo con headers (sin datos adicionales)
    struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) EmptyRecord {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    };
    
    // Raygen record
    EmptyRecord raygen_record_host;
    verificar_optix(
        optixSbtRecordPackHeader(programa_raygen, &raygen_record_host),
        "Pack header raygen"
    );
    
    verificar_cu(cuMemAlloc(&raygen_record, sizeof(EmptyRecord)), "Asignar raygen_record");
    verificar_cuda(
        cudaMemcpy(reinterpret_cast<void*>(raygen_record), &raygen_record_host,
                   sizeof(EmptyRecord), cudaMemcpyHostToDevice),
        "Copiar raygen record"
    );
    
    // Miss record
    EmptyRecord miss_record_host;
    verificar_optix(
        optixSbtRecordPackHeader(programa_miss, &miss_record_host),
        "Pack header miss"
    );
    
    verificar_cu(cuMemAlloc(&miss_record, sizeof(EmptyRecord)), "Asignar miss_record");
    verificar_cuda(
        cudaMemcpy(reinterpret_cast<void*>(miss_record), &miss_record_host,
                   sizeof(EmptyRecord), cudaMemcpyHostToDevice),
        "Copiar miss record"
    );
    
    // Hitgroup record
    EmptyRecord hitgroup_record_host;
    verificar_optix(
        optixSbtRecordPackHeader(programa_hitgroup, &hitgroup_record_host),
        "Pack header hitgroup"
    );
    
    verificar_cu(cuMemAlloc(&hitgroup_record, sizeof(EmptyRecord)), "Asignar hitgroup_record");
    verificar_cuda(
        cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), &hitgroup_record_host,
                   sizeof(EmptyRecord), cudaMemcpyHostToDevice),
        "Copiar hitgroup record"
    );
    
    // Configurar SBT
    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(EmptyRecord);
    sbt.hitgroupRecordCount = 1;
    sbt.exceptionRecord = 0; // Sin manejo de excepciones
    sbt.callablesRecordBase = 0;
    sbt.callablesRecordStrideInBytes = 0;
    sbt.callablesRecordCount = 0;
    
    std::cout << "SBT configurado correctamente" << std::endl;
}

void ArkadeOptiX::construir_gas_con_radio(float radio) {
    const std::vector<Punto3D>& datos_usar = 
        (tipo_distancia == 3) ? datos_normalizados : datos;
    
    // ========================================================================
    // CONSTRUCCION DE AABBs SEGUN EL PAPER ARKADE
    // ========================================================================
    // EQUIVALENCIA MATEMATICA:
    //   Buscar puntos dentro de radio R desde query 
    //   ≡ Query (punto) intersecta AABBs de radio R centrados en dataset
    //
    // Por lo tanto:
    //   - AABBs: Cubos de lado 2*radio centrados en cada punto del dataset
    //   - Rayo: Puntual (longitud ~0) desde la query
    //   - RT Cores retornan AABBs intersectados = CANDIDATOS
    //   - Refine: Verificar geometria exacta (esfera/octaedro/cubo)
    //
    // AABB OCCUPANCY por metrica:
    //   - L2 (esfera): ~52% del AABB contiene puntos validos
    //   - L1 (octaedro): ~33% del AABB contiene puntos validos  
    //   - Linf (cubo): 100% - AABB = geometria exacta
    
    std::vector<OptixAabb> aabbs(datos_usar.size());
    
    for (size_t i = 0; i < datos_usar.size(); i++) {
        const auto& p = datos_usar[i];
        
        // AABB de lado 2*radio centrado en el punto del dataset
        // Para TODAS las metricas usamos el mismo AABB (cubo)
        // La diferencia esta en la fase REFINE donde verificamos geometria exacta
        aabbs[i].minX = p.x - radio;
        aabbs[i].minY = p.y - radio;
        aabbs[i].minZ = p.z - radio;
        aabbs[i].maxX = p.x + radio;
        aabbs[i].maxY = p.y + radio;
        aabbs[i].maxZ = p.z + radio;
    }
    
    // Subir AABBs a GPU
    CUdeviceptr buffer_aabbs_temp;
    size_t tam_aabbs = aabbs.size() * sizeof(OptixAabb);
    verificar_cu(cuMemAlloc(&buffer_aabbs_temp, tam_aabbs), "Asignar buffer_aabbs");
    verificar_cuda(
        cudaMemcpy(reinterpret_cast<void*>(buffer_aabbs_temp), aabbs.data(),
                   tam_aabbs, cudaMemcpyHostToDevice),
        "Copiar AABBs a GPU"
    );
    
    // Configurar build input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    build_input.customPrimitiveArray.aabbBuffers = &buffer_aabbs_temp;
    build_input.customPrimitiveArray.numPrimitives = static_cast<unsigned int>(aabbs.size());
    build_input.customPrimitiveArray.numSbtRecords = 1;
    build_input.customPrimitiveArray.flags = flags;
    
    // Opciones de construcción - RT CORES aceleran esto
    OptixAccelBuildOptions build_options = {};
    build_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    build_options.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    // Calcular memoria
    OptixAccelBufferSizes tam_buffers;
    verificar_optix(
        optixAccelComputeMemoryUsage(contexto_optix, &build_options, &build_input, 1, &tam_buffers),
        "Calcular memoria GAS"
    );
    
    // Liberar GAS anterior si existe
    if (buffer_gas) cuMemFree(buffer_gas);
    
    // Construir GAS (BVH) con RT CORES
    CUdeviceptr buffer_temp_gas;
    verificar_cu(cuMemAlloc(&buffer_temp_gas, tam_buffers.tempSizeInBytes), "Asignar temp_gas");
    verificar_cu(cuMemAlloc(&buffer_gas, tam_buffers.outputSizeInBytes), "Asignar gas");
    
    verificar_optix(
        optixAccelBuild(
            contexto_optix,
            0,
            &build_options,
            &build_input,
            1,
            buffer_temp_gas,
            tam_buffers.tempSizeInBytes,
            buffer_gas,
            tam_buffers.outputSizeInBytes,
            &gas_handle,
            nullptr, 0
        ),
        "Construir GAS"
    );
    
    // Liberar temporales
    cuMemFree(buffer_temp_gas);
    cuMemFree(buffer_aabbs_temp);
    
    std::cout << "GAS (BVH) construido: " << datos_usar.size() << " puntos (handle=" << gas_handle << ")" << std::endl;
}

// buscar_radius deprecado - usar buscar_knn_batch con k grande
std::vector<ResultadoVecino> ArkadeOptiX::buscar_radius(const Punto3D& query, float radio) {
    // Wrapper simple que usa la nueva API de batch
    std::vector<Punto3D> queries_batch = {query};
    auto resultados_batch = buscar_knn_batch(queries_batch, 10000, radio);
    return resultados_batch.empty() ? std::vector<ResultadoVecino>() : resultados_batch[0];
}

std::vector<std::vector<ResultadoVecino>> ArkadeOptiX::buscar_knn_batch(
    const std::vector<Punto3D>& queries, int k, float radio) {
    
    std::vector<std::vector<ResultadoVecino>> resultados;
    resultados.reserve(queries.size());
    
    const std::vector<Punto3D>& datos_usar = 
        (tipo_distancia == 3) ? datos_normalizados : datos;
    
    // ========================================================================
    // CONVERSION DE RADIO PARA DISTANCIA COSENO
    // ========================================================================
    float radio_aabb = radio;
    float radio_refine = radio;
    
    if (tipo_distancia == DIST_COSINE) {
        const float PI = 3.14159265358979323846f;
        float radio_angular = std::min(radio, PI);
        radio_aabb = 2.0f * std::sin(radio_angular / 2.0f);
        radio_refine = radio_angular;
        std::cout << "Coseno: radio_angular=" << radio_angular 
                  << ", radio_aabb=" << radio_aabb << std::endl;
    }
    
    // ========================================================================
    // CONSTRUIR GAS UNA SOLA VEZ (si no existe o radio cambio)
    // ========================================================================
    if (!gas_construido || radio_gas != radio_aabb) {
        std::cout << "Construyendo GAS (BVH) con radio=" << radio_aabb << "..." << std::endl;
        construir_gas_con_radio(radio_aabb);
        gas_construido = true;
        radio_gas = radio_aabb;
    }
    
    // ========================================================================
    // PROCESAR QUERIES EN BATCHES PARALELOS
    // ========================================================================
    // En lugar de procesar 1 query a la vez, procesamos TODAS las queries
    // en paralelo usando optixLaunch con num_queries threads.
    // Cada thread procesa una query diferente.
    
    int num_queries = static_cast<int>(queries.size());
    
    // Reallocate buffers if needed (batch size or k changed)
    if (!buffers_inicializados || num_queries > max_batch_size || k != k_actual) {
        // Liberar buffers anteriores
        if (d_resultados_ids_persistente) cudaFree(d_resultados_ids_persistente);
        if (d_resultados_dists_persistente) cudaFree(d_resultados_dists_persistente);
        if (d_num_resultados_persistente) cudaFree(d_num_resultados_persistente);
        if (d_queries_persistente) cudaFree(d_queries_persistente);
        if (d_params_persistente) cuMemFree(d_params_persistente);
        
        // Asignar nuevos buffers para TODAS las queries
        // Layout: [query0_k_results, query1_k_results, ..., queryN_k_results]
        size_t total_results = num_queries * k;
        verificar_cuda(cudaMalloc(&d_resultados_ids_persistente, total_results * sizeof(int)), "Malloc resultados_ids");
        verificar_cuda(cudaMalloc(&d_resultados_dists_persistente, total_results * sizeof(float)), "Malloc resultados_dists");
        verificar_cuda(cudaMalloc(&d_num_resultados_persistente, num_queries * sizeof(int)), "Malloc num_resultados");
        verificar_cuda(cudaMalloc(&d_queries_persistente, num_queries * sizeof(float3)), "Malloc queries");
        verificar_cu(cuMemAlloc(&d_params_persistente, sizeof(DatosLanzamiento)), "Malloc params");
        
        buffers_inicializados = true;
        max_batch_size = num_queries;
        k_actual = k;
        std::cout << "Buffers para batch inicializados: " << num_queries << " queries x k=" << k << std::endl;
    }
    
    // ========================================================================
    // PREPARAR QUERIES EN GPU
    // ========================================================================
    std::vector<float3> queries_gpu(num_queries);
    for (int i = 0; i < num_queries; i++) {
        Punto3D q = (tipo_distancia == 3) ? queries[i].normalizar() : queries[i];
        queries_gpu[i] = make_float3(q.x, q.y, q.z);
    }
    
    verificar_cuda(
        cudaMemcpy(d_queries_persistente, queries_gpu.data(), 
                   num_queries * sizeof(float3), cudaMemcpyHostToDevice),
        "Copiar queries a GPU"
    );
    
    // Reset contadores de resultados (todos a 0)
    verificar_cuda(
        cudaMemset(d_num_resultados_persistente, 0, num_queries * sizeof(int)),
        "Reset num_resultados"
    );
    
    // ========================================================================
    // CONFIGURAR PARAMETROS PARA BATCH COMPLETO
    // ========================================================================
    DatosLanzamiento params_host;
    params_host.puntos_datos = reinterpret_cast<float3*>(buffer_puntos);
    params_host.ids_datos = reinterpret_cast<int*>(buffer_ids);
    params_host.num_puntos = static_cast<int>(datos_usar.size());
    params_host.queries = d_queries_persistente;
    params_host.num_queries = num_queries;
    params_host.radio = radio_refine;
    params_host.resultados_ids = d_resultados_ids_persistente;
    params_host.resultados_dists = d_resultados_dists_persistente;
    params_host.num_resultados = d_num_resultados_persistente;
    params_host.k = k;
    params_host.tipo_distancia = tipo_distancia;
    params_host.gas_handle = gas_handle;
    
    verificar_cuda(
        cudaMemcpy(reinterpret_cast<void*>(d_params_persistente), &params_host, 
                   sizeof(DatosLanzamiento), cudaMemcpyHostToDevice),
        "Copiar params a GPU"
    );
    
    // ========================================================================
    // LANZAR TODAS LAS QUERIES EN PARALELO!
    // ========================================================================
    // optixLaunch con (num_queries, 1, 1) lanza num_queries threads
    // Cada thread usa optixGetLaunchIndex() para saber cual query procesar
    std::cout << "Lanzando " << num_queries << " queries en paralelo..." << std::endl;
    
    verificar_optix(
        optixLaunch(
            pipeline, 0,
            d_params_persistente,
            sizeof(DatosLanzamiento),
            &sbt,
            num_queries,  // Lanzar num_queries threads en paralelo!
            1, 1
        ),
        "optixLaunch batch"
    );
    
    verificar_cuda(cudaDeviceSynchronize(), "Sync despues de optixLaunch");
    
    // ========================================================================
    // RECUPERAR RESULTADOS DE TODAS LAS QUERIES
    // ========================================================================
    std::vector<int> h_num_resultados(num_queries);
    verificar_cuda(
        cudaMemcpy(h_num_resultados.data(), d_num_resultados_persistente, 
                   num_queries * sizeof(int), cudaMemcpyDeviceToHost),
        "Copiar num_resultados"
    );
    
    // Copiar todos los resultados de una vez
    size_t total_results = num_queries * k;
    std::vector<int> all_ids(total_results);
    std::vector<float> all_dists(total_results);
    
    verificar_cuda(
        cudaMemcpy(all_ids.data(), d_resultados_ids_persistente, 
                   total_results * sizeof(int), cudaMemcpyDeviceToHost),
        "Copiar IDs"
    );
    verificar_cuda(
        cudaMemcpy(all_dists.data(), d_resultados_dists_persistente, 
                   total_results * sizeof(float), cudaMemcpyDeviceToHost),
        "Copiar distancias"
    );
    
    // Organizar resultados por query
    for (int i = 0; i < num_queries; i++) {
        std::vector<ResultadoVecino> res;
        int num_res = std::min(h_num_resultados[i], k);
        res.reserve(num_res);
        
        int offset = i * k;
        for (int j = 0; j < num_res; j++) {
            res.emplace_back(all_ids[offset + j], all_dists[offset + j]);
        }
        
        // Ordenar por distancia (el heap no garantiza orden)
        std::sort(res.begin(), res.end());
        resultados.push_back(res);
    }
    
    std::cout << "Batch completado" << std::endl;
    return resultados;
}

void ArkadeOptiX::limpiar_recursos() {
    // Liberar buffers persistentes
    if (d_resultados_ids_persistente) cudaFree(d_resultados_ids_persistente);
    if (d_resultados_dists_persistente) cudaFree(d_resultados_dists_persistente);
    if (d_num_resultados_persistente) cudaFree(d_num_resultados_persistente);
    if (d_queries_persistente) cudaFree(d_queries_persistente);
    if (d_params_persistente) cuMemFree(d_params_persistente);
    
    if (buffer_puntos) cudaFree(reinterpret_cast<void*>(buffer_puntos));
    if (buffer_ids) cudaFree(reinterpret_cast<void*>(buffer_ids));
    if (buffer_gas) cuMemFree(buffer_gas);
    if (raygen_record) cuMemFree(raygen_record);
    if (miss_record) cuMemFree(miss_record);
    if (hitgroup_record) cuMemFree(hitgroup_record);
    if (pipeline) optixPipelineDestroy(pipeline);
    if (programa_hitgroup) optixProgramGroupDestroy(programa_hitgroup);
    if (programa_miss) optixProgramGroupDestroy(programa_miss);
    if (programa_raygen) optixProgramGroupDestroy(programa_raygen);
    if (modulo_optix) optixModuleDestroy(modulo_optix);
    if (contexto_optix) optixDeviceContextDestroy(contexto_optix);

}

#endif // ARKADE_OPTIX_H
