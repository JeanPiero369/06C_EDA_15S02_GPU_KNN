#ifndef BASELINE_FLANN_H
#define BASELINE_FLANN_H

#include "utilidades.h"
#include <flann/flann.hpp>
#include <vector>
#include <memory>
#include <algorithm>

// ============================================================================
// BASELINE FLANN - CPU Tree-based Search
// Librería: https://github.com/flann-lib/flann
// ============================================================================
//
// Este baseline implementa búsqueda k-NN en CPU usando la biblioteca FLANN
// (Fast Library for Approximate Nearest Neighbors).
//
// MÉTRICAS SOPORTADAS:
//   - L1 (Manhattan): Usa flann::L1<float> nativo
//   - L∞ (Chebyshev): Usa flann::MaxDistance<float> nativo
//
// USO EN EL PAPER:
//   FLANN es el baseline CPU para métricas no-euclidianas (L1, L∞).
//   Para L2 y Coseno, se usa FAISS que es más eficiente.
//
// NOTA IMPORTANTE:
//   FLANN soporta L∞ nativamente con flann::MaxDistance (Chebyshev/L-infinity).
//   NO usar ChiSquareDistance que es una métrica diferente.
//
// ============================================================================

class BaselineFLANN {
private:
    // Índices para cada métrica (solo uno estará activo)
    std::unique_ptr<flann::Index<flann::L1<float>>> indice_l1;
    std::unique_ptr<flann::Index<flann::MaxDistance<float>>> indice_linf;
    
    std::vector<Punto3D> datos;
    std::string tipo_metrica;
    int dimension;
    flann::Matrix<float> dataset_flann;
    
public:
    BaselineFLANN(const std::string& metrica)
        : tipo_metrica(metrica), dimension(3) {
        std::cout << "FLANN CPU inicializado (metrica=" << metrica << ")" << std::endl;
    }
    
    ~BaselineFLANN() {
        if (dataset_flann.ptr()) {
            delete[] dataset_flann.ptr();
        }
    }
    
    void cargar_datos(const std::vector<Punto3D>& puntos) {
        datos = puntos;
        std::cout << "FLANN: " << datos.size() << " puntos cargados" << std::endl;
    }
    
    void construir_indice() {
        std::cout << "Construyendo indice KD-Tree con FLANN..." << std::endl;
        Temporizador timer("Construccion FLANN");
        
        // Convertir datos a formato FLANN (matriz contigua row-major)
        float* datos_array = new float[datos.size() * dimension];
        
        for (size_t i = 0; i < datos.size(); i++) {
            datos_array[i * dimension + 0] = datos[i].x;
            datos_array[i * dimension + 1] = datos[i].y;
            datos_array[i * dimension + 2] = datos[i].z;
        }
        
        dataset_flann = flann::Matrix<float>(datos_array, datos.size(), dimension);
        
        // Parámetros del índice KD-Tree
        // Usamos 4 árboles aleatorios para mejor aproximación
        flann::KDTreeIndexParams params(4);
        
        // Construir índice según métrica
        if (tipo_metrica == "manhattan" || tipo_metrica == "l1") {
            // L1 (Manhattan): sum(|xi - yi|)
            indice_l1 = std::make_unique<flann::Index<flann::L1<float>>>(
                dataset_flann, params
            );
            indice_l1->buildIndex();
            std::cout << "Indice FLANN KD-Tree con L1 (Manhattan) construido" << std::endl;
            
        } else if (tipo_metrica == "chebyshev" || tipo_metrica == "linf") {
            // L∞ (Chebyshev/Max): max(|xi - yi|)
            // FLANN lo soporta nativamente con flann::MaxDistance
            indice_linf = std::make_unique<flann::Index<flann::MaxDistance<float>>>(
                dataset_flann, params
            );
            indice_linf->buildIndex();
            std::cout << "Indice FLANN KD-Tree con L-inf (Chebyshev) construido" << std::endl;
        }
    }
    
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, int k) {
        
        int num_queries = static_cast<int>(queries.size());
        std::vector<std::vector<ResultadoVecino>> resultados;
        resultados.reserve(num_queries);
        
        std::cout << "Ejecutando busqueda k-NN con FLANN (metrica=" << tipo_metrica << ")..." << std::endl;
        Temporizador timer("Busqueda KNN FLANN CPU");
        
        // Preparar queries en formato FLANN (batch processing)
        // Para mejor eficiencia, procesamos todas las queries de una vez
        float* query_array = new float[num_queries * dimension];
        for (int i = 0; i < num_queries; i++) {
            query_array[i * dimension + 0] = queries[i].x;
            query_array[i * dimension + 1] = queries[i].y;
            query_array[i * dimension + 2] = queries[i].z;
        }
        flann::Matrix<float> query_mat(query_array, num_queries, dimension);
        
        // Matrices para resultados
        std::vector<std::vector<int>> indices(num_queries, std::vector<int>(k));
        std::vector<std::vector<float>> dists(num_queries, std::vector<float>(k));
        
        flann::Matrix<int> indices_mat(&indices[0][0], num_queries, k);
        flann::Matrix<float> dists_mat(&dists[0][0], num_queries, k);
        
        // Parámetros de búsqueda
        // checks=128: número de nodos a explorar (trade-off precisión/velocidad)
        flann::SearchParams search_params(128);
        
        // Ejecutar búsqueda según métrica
        if (tipo_metrica == "manhattan" || tipo_metrica == "l1") {
            indice_l1->knnSearch(query_mat, indices_mat, dists_mat, k, search_params);
        } else {
            indice_linf->knnSearch(query_mat, indices_mat, dists_mat, k, search_params);
        }
        
        // Convertir resultados a formato estándar
        for (int i = 0; i < num_queries; i++) {
            std::vector<ResultadoVecino> vecinos;
            vecinos.reserve(k);
            
            for (int j = 0; j < k; j++) {
                int idx = indices[i][j];
                if (idx >= 0 && idx < static_cast<int>(datos.size())) {
                    // FLANN devuelve distancia al cuadrado para algunas métricas
                    // Para L1 y L∞, devuelve la distancia directa
                    float dist = dists[i][j];
                    vecinos.emplace_back(datos[idx].id, dist);
                }
            }
            
            // Ordenar por distancia (FLANN ya debería retornarlos ordenados)
            std::sort(vecinos.begin(), vecinos.end());
            resultados.push_back(vecinos);
            
            if ((i + 1) % 1000 == 0) {
                std::cout << "FLANN: Procesadas " << (i + 1) 
                          << "/" << num_queries << " queries" << std::endl;
            }
        }
        
        delete[] query_array;
        
        return resultados;
    }
};

#endif // BASELINE_FLANN_H
