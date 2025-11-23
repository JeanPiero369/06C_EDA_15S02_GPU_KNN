#ifndef BASELINE_FAISS_H
#define BASELINE_FAISS_H

#include "utilidades.h"
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
// FAISS GPU no disponible en vcpkg build
// #include <faiss/gpu/GpuIndexFlat.h>
// #include <faiss/gpu/StandardGpuResources.h>
#include <vector>
#include <memory>

// ============================================================================
// BASELINE FAISS CPU
// ============================================================================

class BaselineFAISS_CPU {
private:
    std::unique_ptr<faiss::IndexFlatL2> indice_l2;
    std::unique_ptr<faiss::IndexFlat> indice_ip; // Inner product para coseno
    std::vector<Punto3D> datos;
    std::vector<Punto3D> datos_normalizados;
    bool usar_coseno;
    int dimension;
    
public:
    BaselineFAISS_CPU(bool coseno = false) 
        : usar_coseno(coseno), dimension(3) {
        
        if (usar_coseno) {
            // Para coseno: usar inner product sobre vectores normalizados
            indice_ip = std::make_unique<faiss::IndexFlat>(dimension, faiss::METRIC_INNER_PRODUCT);
        } else {
            // Para euclidiana: usar L2
            indice_l2 = std::make_unique<faiss::IndexFlatL2>(dimension);
        }
        
        std::cout << "FAISS CPU inicializado (coseno=" << usar_coseno << ")" << std::endl;
    }
    
    void cargar_datos(const std::vector<Punto3D>& puntos) {
        datos = puntos;
        
        // Preparar datos para FAISS
        std::vector<float> vectores;
        
        if (usar_coseno) {
            // Normalizar vectores
            datos_normalizados.reserve(datos.size());
            for (const auto& p : datos) {
                datos_normalizados.push_back(p.normalizar());
            }
            
            // Convertir a array plano
            vectores.reserve(datos_normalizados.size() * dimension);
            for (const auto& p : datos_normalizados) {
                vectores.push_back(p.x);
                vectores.push_back(p.y);
                vectores.push_back(p.z);
            }
            
            indice_ip->add(datos_normalizados.size(), vectores.data());
        } else {
            // Convertir a array plano
            vectores.reserve(datos.size() * dimension);
            for (const auto& p : datos) {
                vectores.push_back(p.x);
                vectores.push_back(p.y);
                vectores.push_back(p.z);
            }
            
            indice_l2->add(datos.size(), vectores.data());
        }
        
        std::cout << "FAISS CPU: " << datos.size() << " puntos indexados" << std::endl;
    }
    
    std::vector<ResultadoVecino> buscar_knn(const Punto3D& query, int k) {
        // Preparar query
        std::vector<float> query_vec(dimension);
        
        if (usar_coseno) {
            Punto3D query_norm = query.normalizar();
            query_vec[0] = query_norm.x;
            query_vec[1] = query_norm.y;
            query_vec[2] = query_norm.z;
        } else {
            query_vec[0] = query.x;
            query_vec[1] = query.y;
            query_vec[2] = query.z;
        }
        
        // Búsqueda
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distancias(k);
        
        if (usar_coseno) {
            indice_ip->search(1, query_vec.data(), k, distancias.data(), indices.data());
        } else {
            indice_l2->search(1, query_vec.data(), k, distancias.data(), indices.data());
        }
        
        // Convertir resultados
        std::vector<ResultadoVecino> resultados;
        for (int i = 0; i < k; i++) {
            if (indices[i] >= 0 && indices[i] < datos.size()) {
                float dist = distancias[i];
                
                if (usar_coseno) {
                    // FAISS con inner product retorna similitud
                    // Convertir a distancia coseno: dist = 1 - similitud
                    dist = 1.0f - dist;
                } else {
                    // FAISS L2 retorna distancia al cuadrado, tomar raíz
                    dist = std::sqrt(dist);
                }
                
                resultados.emplace_back(datos[indices[i]].id, dist);
            }
        }
        
        return resultados;
    }
    
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, int k) {
        
        std::vector<std::vector<ResultadoVecino>> resultados;
        resultados.reserve(queries.size());
        
        for (size_t i = 0; i < queries.size(); i++) {
            if (i % 1000 == 0) {
                std::cout << "FAISS CPU: Procesando query " << i 
                          << "/" << queries.size() << std::endl;
            }
            resultados.push_back(buscar_knn(queries[i], k));
        }
        
        return resultados;
    }
};

// ============================================================================
// BASELINE FAISS GPU - DESHABILITADO (FAISS vcpkg no incluye GPU support)
// ============================================================================

/*
class BaselineFAISS_GPU {
private:
    std::unique_ptr<faiss::gpu::StandardGpuResources> recursos_gpu;
    std::unique_ptr<faiss::gpu::GpuIndexFlat> indice;
    std::vector<Punto3D> datos;
    std::vector<Punto3D> datos_normalizados;
    bool usar_coseno;
    int dimension;
    
public:
    BaselineFAISS_GPU(bool coseno = false) 
        : usar_coseno(coseno), dimension(3) {
        
        // Crear recursos GPU
        recursos_gpu = std::make_unique<faiss::gpu::StandardGpuResources>();
        
        // Configurar índice
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0; // GPU 0
        
        if (usar_coseno) {
            indice = std::make_unique<faiss::gpu::GpuIndexFlat>(
                recursos_gpu.get(), dimension, faiss::METRIC_INNER_PRODUCT, config);
        } else {
            indice = std::make_unique<faiss::gpu::GpuIndexFlat>(
                recursos_gpu.get(), dimension, faiss::METRIC_L2, config);
        }
        
        std::cout << "FAISS GPU inicializado (coseno=" << usar_coseno << ")" << std::endl;
    }
    
    void cargar_datos(const std::vector<Punto3D>& puntos) {
        datos = puntos;
        
        // Preparar datos para FAISS
        std::vector<float> vectores;
        
        if (usar_coseno) {
            datos_normalizados.reserve(datos.size());
            for (const auto& p : datos) {
                datos_normalizados.push_back(p.normalizar());
            }
            
            vectores.reserve(datos_normalizados.size() * dimension);
            for (const auto& p : datos_normalizados) {
                vectores.push_back(p.x);
                vectores.push_back(p.y);
                vectores.push_back(p.z);
            }
        } else {
            vectores.reserve(datos.size() * dimension);
            for (const auto& p : datos) {
                vectores.push_back(p.x);
                vectores.push_back(p.y);
                vectores.push_back(p.z);
            }
        }
        
        indice->add(vectores.size() / dimension, vectores.data());
        std::cout << "FAISS GPU: " << datos.size() << " puntos indexados" << std::endl;
    }
    
    std::vector<ResultadoVecino> buscar_knn(const Punto3D& query, int k) {
        std::vector<float> query_vec(dimension);
        
        if (usar_coseno) {
            Punto3D query_norm = query.normalizar();
            query_vec[0] = query_norm.x;
            query_vec[1] = query_norm.y;
            query_vec[2] = query_norm.z;
        } else {
            query_vec[0] = query.x;
            query_vec[1] = query.y;
            query_vec[2] = query.z;
        }
        
        std::vector<faiss::idx_t> indices(k);
        std::vector<float> distancias(k);
        
        indice->search(1, query_vec.data(), k, distancias.data(), indices.data());
        
        std::vector<ResultadoVecino> resultados;
        for (int i = 0; i < k; i++) {
            if (indices[i] >= 0 && indices[i] < datos.size()) {
                float dist = distancias[i];
                
                if (usar_coseno) {
                    dist = 1.0f - dist;
                } else {
                    dist = std::sqrt(dist);
                }
                
                resultados.emplace_back(datos[indices[i]].id, dist);
            }
        }
        
        return resultados;
    }
    
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, int k) {
        
        // FAISS GPU puede procesar queries en batch
        std::vector<float> queries_vec;
        queries_vec.reserve(queries.size() * dimension);
        
        if (usar_coseno) {
            for (const auto& q : queries) {
                Punto3D q_norm = q.normalizar();
                queries_vec.push_back(q_norm.x);
                queries_vec.push_back(q_norm.y);
                queries_vec.push_back(q_norm.z);
            }
        } else {
            for (const auto& q : queries) {
                queries_vec.push_back(q.x);
                queries_vec.push_back(q.y);
                queries_vec.push_back(q.z);
            }
        }
        
        std::vector<faiss::idx_t> indices(queries.size() * k);
        std::vector<float> distancias(queries.size() * k);
        
        indice->search(queries.size(), queries_vec.data(), k, 
                      distancias.data(), indices.data());
        
        std::vector<std::vector<ResultadoVecino>> resultados(queries.size());
        
        for (size_t q = 0; q < queries.size(); q++) {
            for (int i = 0; i < k; i++) {
                int idx = q * k + i;
                if (indices[idx] >= 0 && indices[idx] < datos.size()) {
                    float dist = distancias[idx];
                    
                    if (usar_coseno) {
                        dist = 1.0f - dist;
                    } else {
                        dist = std::sqrt(dist);
                    }
                    
                    resultados[q].emplace_back(datos[indices[idx]].id, dist);
                }
            }
        }
        
        return resultados;
    }
};
*/

#endif // BASELINE_FAISS_H
