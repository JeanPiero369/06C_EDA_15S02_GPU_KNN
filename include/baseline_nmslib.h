#ifndef BASELINE_NMSLIB_H
#define BASELINE_NMSLIB_H

#include "utilidades.h"
#include <init.h>
#include <index.h>
#include <knnquery.h>
#include <space.h>
#include <vector>
#include <memory>

// ============================================================================
// BASELINE NMSLIB HNSW
// ============================================================================

class BaselineNMSLIB_HNSW {
private:
    std::unique_ptr<similarity::Index<float>> indice;
    std::unique_ptr<similarity::Space<float>> espacio;
    std::vector<Punto3D> datos;
    std::string tipo_metrica; // "l1" o "linf"
    int dimension;
    
public:
    BaselineNMSLIB_HNSW(const std::string& metrica) 
        : tipo_metrica(metrica), dimension(3) {
        
        // Inicializar librería
        similarity::initLibrary();
        
        // Crear espacio según métrica
        if (metrica == "manhattan" || metrica == "l1") {
            espacio = std::unique_ptr<similarity::Space<float>>(
                similarity::SpaceFactoryRegistry<float>::Instance().
                CreateSpace("l1", similarity::AnyParams())
            );
        } else if (metrica == "chebyshev" || metrica == "linf") {
            espacio = std::unique_ptr<similarity::Space<float>>(
                similarity::SpaceFactoryRegistry<float>::Instance().
                CreateSpace("linf", similarity::AnyParams())
            );
        } else {
            throw std::runtime_error("Metrica no soportada por NMSLIB: " + metrica);
        }
        
        // Crear índice HNSW
        indice = std::unique_ptr<similarity::Index<float>>(
            similarity::MethodFactoryRegistry<float>::Instance().
            CreateMethod(
                false, // print progress
                "hnsw",
                "l2", // método (será sobrescrito por espacio)
                *espacio,
                nullptr // data objects
            )
        );
        
        std::cout << "NMSLIB HNSW inicializado (metrica=" << metrica << ")" << std::endl;
    }
    
    ~BaselineNMSLIB_HNSW() {
        // Limpiar índice antes de destruir espacio
        if (indice) {
            indice.reset();
        }
    }
    
    void cargar_datos(const std::vector<Punto3D>& puntos) {
        datos = puntos;
        
        // Crear objetos de datos para NMSLIB
        similarity::ObjectVector data_objects;
        
        for (size_t i = 0; i < datos.size(); i++) {
            std::vector<float> vec = {datos[i].x, datos[i].y, datos[i].z};
            
            // Crear objeto de datos
            similarity::Object* obj = espacio->CreateObjFromVect(
                i, // ID interno
                -1, // label (no usado)
                vec
            );
            
            data_objects.push_back(obj);
        }
        
        // Agregar datos al índice
        indice->AddData(data_objects);
        
        std::cout << "NMSLIB: " << datos.size() << " puntos agregados" << std::endl;
    }
    
    void construir_indice() {
        // Parámetros HNSW
        similarity::AnyParams params({
            "M=16",              // Número de conexiones por nodo
            "efConstruction=200", // ef durante construcción
            "post=0"             // post processing
        });
        
        std::cout << "Construyendo indice HNSW..." << std::endl;
        Temporizador timer("Construccion HNSW");
        
        indice->CreateIndex(params);
        
        std::cout << "Indice HNSW construido" << std::endl;
    }
    
    std::vector<ResultadoVecino> buscar_knn(const Punto3D& query, int k) {
        // Crear query object
        std::vector<float> query_vec = {query.x, query.y, query.z};
        std::unique_ptr<similarity::Object> query_obj(
            espacio->CreateObjFromVect(0, -1, query_vec)
        );
        
        // Configurar query
        similarity::KNNQuery<float> knn_query(*espacio, query_obj.get(), k);
        
        // Ejecutar búsqueda
        indice->Search(&knn_query);
        
        // Obtener resultados
        std::unique_ptr<similarity::KNNQueue<float>> resultados_nms(
            knn_query.Result()->Clone()
        );
        
        std::vector<ResultadoVecino> resultados;
        
        while (!resultados_nms->Empty()) {
            float dist = resultados_nms->TopDistance();
            size_t idx = resultados_nms->TopObject()->GetId();
            resultados_nms->Pop();
            
            if (idx < datos.size()) {
                resultados.emplace_back(datos[idx].id, dist);
            }
        }
        
        // NMSLIB retorna en orden inverso, revertir
        std::reverse(resultados.begin(), resultados.end());
        
        return resultados;
    }
    
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, int k) {
        
        std::vector<std::vector<ResultadoVecino>> resultados;
        resultados.reserve(queries.size());
        
        for (size_t i = 0; i < queries.size(); i++) {
            if (i % 1000 == 0) {
                std::cout << "NMSLIB: Procesando query " << i 
                          << "/" << queries.size() << std::endl;
            }
            resultados.push_back(buscar_knn(queries[i], k));
        }
        
        return resultados;
    }
};

#endif // BASELINE_NMSLIB_H
