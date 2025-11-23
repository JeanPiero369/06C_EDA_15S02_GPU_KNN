#ifndef BASELINE_FLANN_H
#define BASELINE_FLANN_H

#include "utilidades.h"
#include <flann/flann.hpp>
#include <vector>
#include <memory>

// ============================================================================
// BASELINE FLANN - CPU Tree-based Search
// Librería: https://github.com/flann-lib/flann
// ============================================================================

class BaselineFLANN {
private:
    std::unique_ptr<flann::Index<flann::L1<float>>> indice_l1;
    std::unique_ptr<flann::Index<flann::ChiSquareDistance<float>>> indice_linf;
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
        
        // Convertir datos a formato Treelogy (array plano)
        // Convertir datos a formato FLANN (matriz contigua)
        float* datos_array = new float[datos.size() * dimension];
        
        for (size_t i = 0; i < datos.size(); i++) {
            datos_array[i * dimension + 0] = datos[i].x;
            datos_array[i * dimension + 1] = datos[i].y;
            datos_array[i * dimension + 2] = datos[i].z;
        }
        
        dataset_flann = flann::Matrix<float>(datos_array, datos.size(), dimension);
        
        // Parámetros del índice KD-Tree
        flann::KDTreeIndexParams params(4); // 4 KD-Trees aleatorios
        
        // Construir índice según métrica
        if (tipo_metrica == "manhattan" || tipo_metrica == "l1") {
            indice_l1 = std::make_unique<flann::Index<flann::L1<float>>>(
                dataset_flann, params
            );
            indice_l1->buildIndex();
        } else if (tipo_metrica == "chebyshev" || tipo_metrica == "linf") {
            // FLANN no tiene L-infinito nativo, usar distancia máxima customizada
            // Usamos ChiSquare como aproximación para este caso
            indice_linf = std::make_unique<flann::Index<flann::ChiSquareDistance<float>>>(
                dataset_flann, params
            );
            indice_linf->buildIndex();
        }
        
        std::cout << "Indice FLANN KD-Tree construido" << std::endl;
    }
    
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, int k) {
        
        int num_queries = queries.size();
        std::vector<std::vector<ResultadoVecino>> resultados;
        resultados.reserve(num_queries);
        
        std::cout << "Ejecutando busqueda k-NN con FLANN..." << std::endl;
        Temporizador timer("Busqueda KNN FLANN CPU");
        
        // Preparar queries en formato FLANN
        float* query_array = new float[dimension];
        std::vector<std::vector<int>> indices(1, std::vector<int>(k));
        std::vector<std::vector<float>> dists(1, std::vector<float>(k));
        
        flann::Matrix<int> indices_mat(&indices[0][0], 1, k);
        flann::Matrix<float> dists_mat(&dists[0][0], 1, k);
        flann::SearchParams search_params(128); // Checks = 128
        
        for (int i = 0; i < num_queries; i++) {
            query_array[0] = queries[i].x;
            query_array[1] = queries[i].y;
            query_array[2] = queries[i].z;
            
            flann::Matrix<float> query_mat(query_array, 1, dimension);
            
            // Buscar según métrica
            if (tipo_metrica == "manhattan" || tipo_metrica == "l1") {
                indice_l1->knnSearch(query_mat, indices_mat, dists_mat, k, search_params);
            } else {
                // Para Chebyshev, usar búsqueda y recalcular distancias manualmente
                indice_linf->knnSearch(query_mat, indices_mat, dists_mat, k, search_params);
                
                // Recalcular distancias L-infinito
                for (int j = 0; j < k; j++) {
                    int idx = indices[0][j];
                    if (idx >= 0 && idx < (int)datos.size()) {
                        dists[0][j] = distancia_chebyshev(queries[i], datos[idx]);
                    }
                }
            }
            
            // Convertir resultados
            std::vector<ResultadoVecino> vecinos;
            vecinos.reserve(k);
            
            for (int j = 0; j < k; j++) {
                int idx = indices[0][j];
                if (idx >= 0 && idx < (int)datos.size()) {
                    vecinos.emplace_back(datos[idx].id, dists[0][j]);
                }
            }
            
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
