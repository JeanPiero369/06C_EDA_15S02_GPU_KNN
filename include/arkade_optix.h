#ifndef ARKADE_OPTIX_H
#define ARKADE_OPTIX_H

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
    float3* puntos_datos;
    int* ids_datos;
    int num_puntos;
    float3 query;
    float radio;
    int* resultados_ids;
    float* resultados_dists;
    int* num_resultados;
    int tipo_distancia; // 0=L2, 1=L1, 2=Linf, 3=Coseno
    OptixTraversableHandle gas_handle; // Handle del GAS
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
    ArkadeOptiX(int tipo_dist) : tipo_distancia(tipo_dist) {
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
        const std::vector<Punto3D>& queries, int k);
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
    
    // Hitgroup program
    OptixProgramGroupDesc desc_hitgroup = {};
    desc_hitgroup.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc_hitgroup.hitgroup.moduleCH = modulo_optix;
    desc_hitgroup.hitgroup.entryFunctionNameCH = "__closesthit__ch";
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
    
    // Crear AABBs que encierren las formas geométricas según la métrica
    std::vector<OptixAabb> aabbs(datos_usar.size());
    
    for (size_t i = 0; i < datos_usar.size(); i++) {
        const auto& p = datos_usar[i];
        
        switch(tipo_distancia) {
            case 0: // L2 (Euclidean) - ESFERA de radio r
                aabbs[i].minX = p.x - radio;
                aabbs[i].minY = p.y - radio;
                aabbs[i].minZ = p.z - radio;
                aabbs[i].maxX = p.x + radio;
                aabbs[i].maxY = p.y + radio;
                aabbs[i].maxZ = p.z + radio;
                break;
                
            case 1: // L1 (Manhattan) - OCTAEDRO/ROMBO de radio r
                aabbs[i].minX = p.x - radio;
                aabbs[i].minY = p.y - radio;
                aabbs[i].minZ = p.z - radio;
                aabbs[i].maxX = p.x + radio;
                aabbs[i].maxY = p.y + radio;
                aabbs[i].maxZ = p.z + radio;
                break;
                
            case 2: // L∞ (Chebyshev) - CUBO de radio r
                aabbs[i].minX = p.x - radio;
                aabbs[i].minY = p.y - radio;
                aabbs[i].minZ = p.z - radio;
                aabbs[i].maxX = p.x + radio;
                aabbs[i].maxY = p.y + radio;
                aabbs[i].maxZ = p.z + radio;
                break;
                
            case 3: // Coseno - ESFERA en espacio normalizado
                aabbs[i].minX = p.x - radio;
                aabbs[i].minY = p.y - radio;
                aabbs[i].minZ = p.z - radio;
                aabbs[i].maxX = p.x + radio;
                aabbs[i].maxY = p.y + radio;
                aabbs[i].maxZ = p.z + radio;
                break;
        }
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
    
    // Opciones de construcción
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
    
    // Construir GAS con RT CORES
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
    
    std::cout << "GAS construido con radio " << radio << " (handle=" << gas_handle << ")" << std::endl;
}

std::vector<ResultadoVecino> ArkadeOptiX::buscar_radius(const Punto3D& query, float radio) {
    std::vector<ResultadoVecino> resultados;
    
    const std::vector<Punto3D>& datos_usar = 
        (tipo_distancia == 3) ? datos_normalizados : datos;
    Punto3D query_usar = (tipo_distancia == 3) ? query.normalizar() : query;
    
    // ========================================================================
    // USAR GAS PRE-CONSTRUIDO (construido una vez en buscar_knn_batch)
    // ========================================================================
    
    const int MAX_RESULTADOS = 10000;
    int* d_resultados_ids;
    float* d_resultados_dists;
    int* d_num_resultados;
    int h_num_resultados = 0;
    
    verificar_cuda(cudaMalloc(&d_resultados_ids, MAX_RESULTADOS * sizeof(int)), "Malloc resultados_ids");
    verificar_cuda(cudaMalloc(&d_resultados_dists, MAX_RESULTADOS * sizeof(float)), "Malloc resultados_dists");
    verificar_cuda(cudaMalloc(&d_num_resultados, sizeof(int)), "Malloc num_resultados");
    verificar_cuda(cudaMemcpy(d_num_resultados, &h_num_resultados, sizeof(int), cudaMemcpyHostToDevice), "Copiar num_resultados inicial");
    
    // ========================================================================
    // FASE 3: CONFIGURAR PARÁMETROS Y LANZAR RT CORES
    // ========================================================================
    
    DatosLanzamiento params_host;
    params_host.puntos_datos = reinterpret_cast<float3*>(buffer_puntos);
    params_host.ids_datos = reinterpret_cast<int*>(buffer_ids);
    params_host.num_puntos = static_cast<int>(datos_usar.size());
    params_host.query = make_float3(query_usar.x, query_usar.y, query_usar.z);
    params_host.radio = radio;
    params_host.resultados_ids = d_resultados_ids;
    params_host.resultados_dists = d_resultados_dists;
    params_host.num_resultados = d_num_resultados;
    params_host.tipo_distancia = tipo_distancia;
    params_host.gas_handle = gas_handle; // Usar GAS pre-construido global
    
    CUdeviceptr d_params;
    verificar_cu(cuMemAlloc(&d_params, sizeof(DatosLanzamiento)), "Malloc params");
    verificar_cuda(
        cudaMemcpy(reinterpret_cast<void*>(d_params), &params_host, 
                   sizeof(DatosLanzamiento), cudaMemcpyHostToDevice),
        "Copiar params a GPU"
    );
    
    // Lanzar OptiX - procesar todos los puntos en paralelo con RT Cores
    int num_puntos = static_cast<int>(datos_usar.size());
    verificar_optix(
        optixLaunch(
            pipeline,
            0,
            d_params,
            sizeof(DatosLanzamiento),
            &sbt,
            num_puntos, // Un thread por punto para filtrado paralelo
            1,
            1
        ),
        "optixLaunch"
    );
    
    verificar_cuda(cudaDeviceSynchronize(), "Sincronizar después de optixLaunch");
    
    // ========================================================================
    // FASE 4: RECUPERAR RESULTADOS
    // ========================================================================
    
    verificar_cuda(
        cudaMemcpy(&h_num_resultados, d_num_resultados, sizeof(int), cudaMemcpyDeviceToHost),
        "Copiar num_resultados desde GPU"
    );
    
    if (h_num_resultados > 0) {
        int num_real = std::min(h_num_resultados, MAX_RESULTADOS);
        std::vector<int> ids_host(num_real);
        std::vector<float> dists_host(num_real);
        
        verificar_cuda(
            cudaMemcpy(ids_host.data(), d_resultados_ids, num_real * sizeof(int), cudaMemcpyDeviceToHost),
            "Copiar IDs desde GPU"
        );
        verificar_cuda(
            cudaMemcpy(dists_host.data(), d_resultados_dists, num_real * sizeof(float), cudaMemcpyDeviceToHost),
            "Copiar distancias desde GPU"
        );
        
        for (int i = 0; i < num_real; i++) {
            resultados.emplace_back(ids_host[i], dists_host[i]);
        }
    }
    
    // Limpiar
    cudaFree(d_resultados_ids);
    cudaFree(d_resultados_dists);
    cudaFree(d_num_resultados);
    cuMemFree(d_params);
    
    std::sort(resultados.begin(), resultados.end());
    return resultados;
}

std::vector<std::vector<ResultadoVecino>> ArkadeOptiX::buscar_knn_batch(
    const std::vector<Punto3D>& queries, int k) {
    
    std::vector<std::vector<ResultadoVecino>> resultados;
    resultados.reserve(queries.size());
    
    // Radio inicial
    float radio = 50.0f;
    
    // CONSTRUIR GAS UNA SOLA VEZ para todas las queries
    std::cout << "Construyendo GAS una vez para " << queries.size() << " queries..." << std::endl;
    construir_gas_con_radio(radio);
    
    // Procesar todas las queries con el mismo GAS
    for (size_t i = 0; i < queries.size(); i++) {
        if (i % 1000 == 0) {
            std::cout << "Procesando query " << i << "/" << queries.size() << std::endl;
        }
        
        auto res = buscar_radius(queries[i], radio);
        
        // Si no hay suficientes resultados, reconstruir GAS con radio mayor
        if (res.size() < k) {
            radio *= 2.0f;
            std::cout << "Radio insuficiente, reconstruyendo GAS con radio=" << radio << std::endl;
            construir_gas_con_radio(radio);
            res = buscar_radius(queries[i], radio);
        }
        
        if (res.size() > k) {
            res.resize(k);
        }
        
        resultados.push_back(res);
    }
    
    return resultados;
}

void ArkadeOptiX::limpiar_recursos() {
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
