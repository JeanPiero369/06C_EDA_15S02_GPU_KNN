#ifndef BASELINE_FASTRNN_H
#define BASELINE_FASTRNN_H

#include "arkade_optix.h"
#include <cmath>

/**
 * FastRNN Baseline - GPU implementation using RT Cores
 * 
 * Basado en el paper FastRNN que usa RT architecture para fixed-radius search.
 * Para distancias no-Euclidianas (L1, L∞), se usa:
 * - Radio expandido: √(d*r) donde d=dimensión, r=radio original
 * - k' > k para compensar y obtener k vecinos correctos según la distancia
 */
class BaselineFastRNN {
private:
    int tipo_distancia;
    ArkadeOptiX* arkade;
    int dimensiones;
    
public:
    BaselineFastRNN(const std::string& metrica, int dim = 3) 
        : dimensiones(dim), arkade(nullptr) {
        if (metrica == "manhattan" || metrica == "l1") {
            tipo_distancia = ArkadeOptiX::DIST_L1;
        } else if (metrica == "chebyshev" || metrica == "linf") {
            tipo_distancia = ArkadeOptiX::DIST_LINF;
        } else {
            throw std::runtime_error("FastRNN solo soporta métricas L1 y L∞");
        }
        arkade = new ArkadeOptiX(tipo_distancia);
    }
    
    ~BaselineFastRNN() {
        delete arkade;
    }
    
    void cargar_datos(const std::vector<Punto3D>& datos) {
        arkade->cargar_datos(datos);
    }
    
    void construir_indice() {
        arkade->inicializar_optix();
        arkade->construir_gas();
    }
    
    /**
     * Búsqueda kNN usando estrategia FastRNN:
     * - Radio expandido para capturar suficientes candidatos
     * - FastRNN paper: usa √(d*r) para capturar vecinos en métricas no-Euclidianas
     */
    std::vector<std::vector<ResultadoVecino>> buscar_knn_batch(
        const std::vector<Punto3D>& queries, 
        int k
    ) {
        // FastRNN: la implementación en arkade ya maneja el radio expandido
        // internamente para cada métrica
        return arkade->buscar_knn_batch(queries, k);
    }
};

#endif // BASELINE_FASTRNN_H
