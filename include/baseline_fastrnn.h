#ifndef BASELINE_FASTRNN_H
#define BASELINE_FASTRNN_H

#include "arkade_optix.h"
#include <cmath>

/**
 * ============================================================================
 * FastRNN GPU Baseline - Implementacion RT Cores SIN optimizacion batch
 * ============================================================================
 * 
 * NOTA IMPORTANTE: Esta NO es la libreria FastRNN original.
 * Es una implementacion que simula el enfoque FastRNN usando OptiX RT Cores
 * pero SIN la optimizacion de construccion batch del GAS.
 * 
 * DIFERENCIA CON ARKADE:
 * - FastRNN (este baseline): Construye GAS para cada query individualmente
 * - Arkade: Construye GAS UNA VEZ y lo reutiliza para todas las queries
 * 
 * Basado en el paper FastRNN que usa RT architecture para fixed-radius search.
 * Para distancias no-Euclidianas (L1, Linf), se usa radio expandido.
 * 
 * METRICAS SOPORTADAS:
 * - L1 (Manhattan): Geometria octaedro/bipiramide
 * - Linf (Chebyshev): Geometria cubo (100% ocupacion AABB)
 */
class BaselineFastRNN {
private:
    int tipo_distancia;
    ArkadeOptiX* arkade;
    int dimensiones;
    std::vector<Punto3D> datos_almacenados;
    
public:
    BaselineFastRNN(const std::string& metrica, int dim = 3) 
        : dimensiones(dim), arkade(nullptr) {
        if (metrica == "manhattan" || metrica == "l1") {
            tipo_distancia = ArkadeOptiX::DIST_L1;
        } else if (metrica == "chebyshev" || metrica == "linf") {
            tipo_distancia = ArkadeOptiX::DIST_LINF;
        } else {
            throw std::runtime_error("FastRNN solo soporta metricas L1 y Linf");
        }
        arkade = new ArkadeOptiX(tipo_distancia);
    }
    
    ~BaselineFastRNN() {
        delete arkade;
    }
    
    void cargar_datos(const std::vector<Punto3D>& datos) {
        datos_almacenados = datos;
        arkade->cargar_datos(datos);
    }
    
    void construir_indice() {
        arkade->inicializar_optix();
        arkade->construir_gas();
    }
    
    /**
     * Busqueda radius query usando estrategia FastRNN (sin batch GAS)
     * 
     * A diferencia de Arkade que construye el GAS una sola vez,
     * FastRNN baseline reconstruye para simular el enfoque sin optimizacion.
     */
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, 
        int k,
        float radio = 100.0f
    ) {
        std::vector<std::vector<ResultadoVecino>> resultados;
        resultados.reserve(queries.size());
        
        // FastRNN baseline: Construir GAS para cada query (sin optimizacion batch)
        // Esto es mas lento que Arkade pero representa el enfoque baseline
        std::cout << "FastRNN baseline: Procesando " << queries.size() << " queries (sin batch GAS)..." << std::endl;
        
        for (size_t i = 0; i < queries.size(); i++) {
            if (i % 1000 == 0) {
                std::cout << "FastRNN query " << i << "/" << queries.size() << std::endl;
            }
            
            // Reconstruir GAS para cada query (comportamiento baseline sin optimizacion)
            arkade->construir_gas_con_radio(radio);
            
            auto res = arkade->buscar_radius(queries[i], radio);
            
            std::sort(res.begin(), res.end());
            if (res.size() > static_cast<size_t>(k)) {
                res.resize(k);
            }
            
            resultados.push_back(res);
        }
        
        return resultados;
    }
};

#endif // BASELINE_FASTRNN_H
