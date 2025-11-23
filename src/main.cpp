#define NOMINMAX // Evitar conflictos con min/max en Windows
#include "utilidades.h"
#include "arkade_optix.h"
#include "baseline_faiss.h"
#include "baseline_fastrnn.h"
#include "baseline_flann.h"
#include <iostream>
#include <iomanip>
#include <algorithm> // Para std::min, std::max, std::set_intersection

// ============================================================================
// CONFIGURACIÓN GLOBAL
// ============================================================================

const int K = 10; // Número de vecinos a buscar
const std::string RUTA_DATOS = "data/data.csv";
const std::string RUTA_QUERIES = "data/queries.csv";
const std::string RUTA_GROUND_TRUTH_L2 = "data/knn_euclidean.csv";
const std::string RUTA_GROUND_TRUTH_L1 = "data/knn_manhattan.csv";
const std::string RUTA_GROUND_TRUTH_LINF = "data/knn_chebyshev.csv";
const std::string RUTA_GROUND_TRUTH_COSINE = "data/knn_cosine.csv";

// ============================================================================
// FUNCIÓN PARA VALIDAR RESULTADOS
// ============================================================================

struct MetricasValidacion {
    double precision_promedio;
    double recall_promedio;
    double error_distancia_promedio;
    int queries_correctas;
    int total_queries;
};

// Convertir ResultadosKNN a formato estándar para validación
std::vector<std::vector<ResultadoVecino>> convertir_resultados_knn(const CargadorCSV::ResultadosKNN& knn) {
    std::vector<std::vector<ResultadoVecino>> resultado;
    resultado.reserve(knn.vecinos_ids.size());
    
    for (size_t i = 0; i < knn.vecinos_ids.size(); i++) {
        std::vector<ResultadoVecino> vecinos;
        vecinos.reserve(knn.vecinos_ids[i].size());
        
        for (size_t j = 0; j < knn.vecinos_ids[i].size(); j++) {
            ResultadoVecino v;
            v.id = knn.vecinos_ids[i][j];
            v.distancia = knn.vecinos_dists[i][j];
            vecinos.push_back(v);
        }
        resultado.push_back(vecinos);
    }
    return resultado;
}

MetricasValidacion validar_resultados(
    const std::vector<std::vector<ResultadoVecino>>& resultados,
    const std::vector<std::vector<ResultadoVecino>>& ground_truth,
    int k
) {
    MetricasValidacion metricas = {0.0, 0.0, 0.0, 0, 0};
    
    if (resultados.size() != ground_truth.size()) {
        std::cerr << "Error: tamaños diferentes" << std::endl;
        return metricas;
    }
    
    metricas.total_queries = resultados.size();
    
    for (size_t i = 0; i < resultados.size(); i++) {
        const auto& res = resultados[i];
        const auto& gt = ground_truth[i];
        
        // Crear sets de IDs para comparación
        std::set<int> ids_res;
        std::set<int> ids_gt;
        
        for (int j = 0; j < std::min(k, (int)res.size()); j++) {
            ids_res.insert(res[j].id);
        }
        
        for (int j = 0; j < std::min(k, (int)gt.size()); j++) {
            ids_gt.insert(gt[j].id);
        }
        
        // Calcular intersección
        std::vector<int> interseccion;
        std::set_intersection(
            ids_res.begin(), ids_res.end(),
            ids_gt.begin(), ids_gt.end(),
            std::back_inserter(interseccion)
        );
        
        // Precision y Recall
        double precision = ids_res.empty() ? 0.0 : (double)interseccion.size() / ids_res.size();
        double recall = ids_gt.empty() ? 0.0 : (double)interseccion.size() / ids_gt.size();
        
        metricas.precision_promedio += precision;
        metricas.recall_promedio += recall;
        
        // Error de distancia (promedio de diferencias)
        double error_dist = 0.0;
        int num_comparaciones = std::min(res.size(), gt.size());
        for (int j = 0; j < num_comparaciones; j++) {
            error_dist += std::abs(res[j].distancia - gt[j].distancia);
        }
        metricas.error_distancia_promedio += (num_comparaciones > 0) ? error_dist / num_comparaciones : 0.0;
        
        // Query correcta si recall >= 0.9
        if (recall >= 0.9) {
            metricas.queries_correctas++;
        }
    }
    
    metricas.precision_promedio /= metricas.total_queries;
    metricas.recall_promedio /= metricas.total_queries;
    metricas.error_distancia_promedio /= metricas.total_queries;
    
    return metricas;
}

// ============================================================================
// FUNCIÓN PARA IMPRIMIR TABLA DE COMPARACIÓN
// ============================================================================

void imprimir_tabla_comparacion(
    const std::string& metrica,
    const std::vector<std::tuple<std::string, double, MetricasValidacion>>& resultados_metodos
) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "COMPARACIÓN: " << metrica << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << std::setw(25) << "Método"
              << std::setw(15) << "Tiempo (ms)"
              << std::setw(15) << "Exactitud (%)"
              << std::setw(12) << "Precision"
              << std::setw(12) << "Recall" << std::endl;
    std::cout << std::string(79, '-') << std::endl;
    
    for (const auto& [metodo, tiempo_ms, metricas] : resultados_metodos) {
        double exactitud = (double)metricas.queries_correctas / metricas.total_queries * 100.0;
        
        std::cout << std::setw(25) << metodo
                  << std::setw(15) << std::fixed << std::setprecision(2) << tiempo_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << exactitud
                  << std::setw(12) << std::fixed << std::setprecision(4) << metricas.precision_promedio
                  << std::setw(12) << std::fixed << std::setprecision(4) << metricas.recall_promedio << std::endl;
    }
    
    std::cout << std::string(79, '=') << std::endl;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "ARKADE: k-NN Search con GPU Ray Tracing" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    try {
        // ====================================================================
        // 1. CARGAR DATOS
        // ====================================================================
        
        std::cout << "Cargando datos..." << std::endl;
        auto datos = CargadorCSV::cargar_datos(RUTA_DATOS);
        auto queries = CargadorCSV::cargar_queries(RUTA_QUERIES);
        std::cout << "Datos: " << datos.size() << " puntos" << std::endl;
        std::cout << "Queries: " << queries.size() << " consultas\n" << std::endl;
        
        // Cargar ground truth y convertir a formato estándar
        auto gt_l2 = convertir_resultados_knn(CargadorCSV::cargar_resultados_knn(RUTA_GROUND_TRUTH_L2));
        auto gt_l1 = convertir_resultados_knn(CargadorCSV::cargar_resultados_knn(RUTA_GROUND_TRUTH_L1));
        auto gt_linf = convertir_resultados_knn(CargadorCSV::cargar_resultados_knn(RUTA_GROUND_TRUTH_LINF));
        auto gt_cosine = convertir_resultados_knn(CargadorCSV::cargar_resultados_knn(RUTA_GROUND_TRUTH_COSINE));
        
        // ====================================================================
        // 2. EXPERIMENTOS CON L2 (EUCLIDEAN)
        // ====================================================================
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "DISTANCIA L2 (EUCLIDEAN)" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::vector<std::tuple<std::string, double, MetricasValidacion>> resultados_l2;
        
        // Arkade OptiX L2
        {
            std::cout << "--- Arkade OptiX L2 ---" << std::endl;
            ArkadeOptiX arkade(ArkadeOptiX::DIST_L2);
            arkade.inicializar_optix();
            arkade.cargar_datos(datos);
            arkade.construir_gas();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = arkade.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "Arkade OptiX L2 (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/ARKADE_knn_euclidean.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_l2, K);
            resultados_l2.emplace_back("Arkade OptiX", tiempo_ms, metricas);
        }
        
        // FAISS CPU L2
        {
            std::cout << "\n--- FAISS CPU L2 ---" << std::endl;
            BaselineFAISS_CPU faiss_cpu(false);
            faiss_cpu.cargar_datos(datos);
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = faiss_cpu.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "FAISS CPU L2 (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/FAISS_CPU_knn_euclidean.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_l2, K);
            resultados_l2.emplace_back("FAISS CPU", tiempo_ms, metricas);
        }
        
        // FAISS GPU L2 - DESHABILITADO (vcpkg FAISS sin GPU support)
        /*
        {
            std::cout << "\n--- FAISS GPU L2 ---" << std::endl;
            BaselineFAISS_GPU faiss_gpu("euclidean");
            faiss_gpu.cargar_datos(datos);
            faiss_gpu.construir_indice();
            
            Temporizador timer("FAISS GPU L2 (busqueda)");
            auto resultados = faiss_gpu.buscar_knn_batch(queries, K);
            
            ExportadorResultados::guardar_resultados_knn("results/FAISS_GPU_knn_euclidean.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_l2, K);
            resultados_l2.emplace_back("FAISS GPU", metricas);
        }
        */
        
        imprimir_tabla_comparacion("L2 (Euclidean)", resultados_l2);
        
        // ====================================================================
        // 3. EXPERIMENTOS CON L1 (MANHATTAN)
        // ====================================================================
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "DISTANCIA L1 (MANHATTAN)" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::vector<std::tuple<std::string, double, MetricasValidacion>> resultados_l1;
        
        // Arkade OptiX L1
        {
            std::cout << "--- Arkade OptiX L1 ---" << std::endl;
            ArkadeOptiX arkade(ArkadeOptiX::DIST_L1);
            arkade.inicializar_optix();
            arkade.cargar_datos(datos);
            arkade.construir_gas();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = arkade.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "Arkade OptiX L1 (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/ARKADE_knn_manhattan.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_l1, K);
            resultados_l1.emplace_back("Arkade OptiX", tiempo_ms, metricas);
        }
        
        // FastRNN L1 (GPU Baseline - usa RT Cores con ajuste para Manhattan)
        {
            std::cout << "\n--- FastRNN GPU L1 (GPU Baseline) ---" << std::endl;
            BaselineFastRNN fastrnn("manhattan");
            fastrnn.cargar_datos(datos);
            fastrnn.construir_indice();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = fastrnn.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "FastRNN GPU L1 (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/FastRNN_GPU_knn_manhattan.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_l1, K);
            resultados_l1.emplace_back("FastRNN GPU", tiempo_ms, metricas);
        }
        
        // FLANN CPU L1 (CPU Baseline)
        {
            std::cout << "\n--- FLANN CPU L1 (CPU Baseline) ---" << std::endl;
            BaselineFLANN flann("manhattan");
            flann.cargar_datos(datos);
            flann.construir_indice();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = flann.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "FLANN CPU L1 (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/FLANN_GPU_knn_manhattan.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_l1, K);
            resultados_l1.emplace_back("FLANN CPU (GPU Baseline)", tiempo_ms, metricas);
        }
        
        imprimir_tabla_comparacion("L1 (Manhattan)", resultados_l1);
        
        // ====================================================================
        // 4. EXPERIMENTOS CON L∞ (CHEBYSHEV)
        // ====================================================================
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "DISTANCIA L∞ (CHEBYSHEV)" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::vector<std::tuple<std::string, double, MetricasValidacion>> resultados_linf;
        
        // Arkade OptiX L∞
        {
            std::cout << "--- Arkade OptiX L∞ ---" << std::endl;
            ArkadeOptiX arkade(ArkadeOptiX::DIST_LINF);
            arkade.inicializar_optix();
            arkade.cargar_datos(datos);
            arkade.construir_gas();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = arkade.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "Arkade OptiX L∞ (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/ARKADE_knn_chebyshev.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_linf, K);
            resultados_linf.emplace_back("Arkade OptiX", tiempo_ms, metricas);
        }
        
        // FastRNN L∞ (GPU Baseline - usa RT Cores con ajuste para Chebyshev)
        {
            std::cout << "\n--- FastRNN GPU L∞ (GPU Baseline) ---" << std::endl;
            BaselineFastRNN fastrnn("chebyshev");
            fastrnn.cargar_datos(datos);
            fastrnn.construir_indice();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = fastrnn.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "FastRNN GPU L∞ (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/FastRNN_GPU_knn_chebyshev.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_linf, K);
            resultados_linf.emplace_back("FastRNN GPU", tiempo_ms, metricas);
        }
        
        // FLANN CPU L∞ (CPU Baseline)
        {
            std::cout << "\n--- FLANN CPU L∞ (CPU Baseline) ---" << std::endl;
            BaselineFLANN flann("chebyshev");
            flann.cargar_datos(datos);
            flann.construir_indice();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = flann.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "FLANN CPU L∞ (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/FLANN_GPU_knn_chebyshev.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_linf, K);
            resultados_linf.emplace_back("FLANN CPU (GPU Baseline)", tiempo_ms, metricas);
        }
        
        imprimir_tabla_comparacion("L∞ (Chebyshev)", resultados_linf);
        
        // ====================================================================
        // 5. EXPERIMENTOS CON COSINE
        // ====================================================================
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "DISTANCIA COSINE" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        std::vector<std::tuple<std::string, double, MetricasValidacion>> resultados_cosine;
        
        // Arkade OptiX Cosine
        {
            std::cout << "--- Arkade OptiX Cosine ---" << std::endl;
            ArkadeOptiX arkade(ArkadeOptiX::DIST_COSINE);
            arkade.inicializar_optix();
            arkade.cargar_datos(datos);
            arkade.construir_gas();
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = arkade.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "Arkade OptiX Cosine (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/ARKADE_knn_cosine.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_cosine, K);
            resultados_cosine.emplace_back("Arkade OptiX", tiempo_ms, metricas);
        }
        
        // FAISS CPU Cosine
        {
            std::cout << "\n--- FAISS CPU Cosine ---" << std::endl;
            BaselineFAISS_CPU faiss_cpu(true);
            faiss_cpu.cargar_datos(datos);
            
            auto inicio = std::chrono::high_resolution_clock::now();
            auto resultados = faiss_cpu.buscar_knn_batch(queries, K);
            auto fin = std::chrono::high_resolution_clock::now();
            double tiempo_ms = std::chrono::duration<double, std::milli>(fin - inicio).count();
            std::cout << "FAISS CPU Cosine (busqueda): " << tiempo_ms << " ms" << std::endl;
            
            ExportadorResultados::guardar_resultados_knn("results/FAISS_CPU_knn_cosine.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_cosine, K);
            resultados_cosine.emplace_back("FAISS CPU", tiempo_ms, metricas);
        }
        
        // FAISS GPU Cosine - DESHABILITADO (vcpkg FAISS sin GPU support)
        /*
        {
            std::cout << "\n--- FAISS GPU Cosine ---" << std::endl;
            BaselineFAISS_GPU faiss_gpu("cosine");
            faiss_gpu.cargar_datos(datos);
            faiss_gpu.construir_indice();
            
            Temporizador timer("FAISS GPU Cosine (busqueda)");
            auto resultados = faiss_gpu.buscar_knn_batch(queries, K);
            
            ExportadorResultados::guardar_resultados_knn("results/FAISS_GPU_knn_cosine.csv", resultados);
            auto metricas = validar_resultados(resultados, gt_cosine, K);
            resultados_cosine.emplace_back("FAISS GPU", metricas);
        }
        */
        
        imprimir_tabla_comparacion("Cosine", resultados_cosine);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "EXPERIMENTOS COMPLETADOS" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Resultados guardados en directorio results/" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
