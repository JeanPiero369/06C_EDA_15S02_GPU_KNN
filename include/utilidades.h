#ifndef UTILIDADES_H
#define UTILIDADES_H

#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// ============================================================================
// ESTRUCTURAS BASICAS
// ============================================================================

struct Punto3D {
    float x, y, z;
    int id;
    
    Punto3D() : x(0), y(0), z(0), id(-1) {}
    Punto3D(float x_, float y_, float z_, int id_ = -1) 
        : x(x_), y(y_), z(z_), id(id_) {}
    
    Punto3D normalizar() const {
        float magnitud = std::sqrt(x*x + y*y + z*z);
        if (magnitud < 1e-10f) return Punto3D(0, 0, 0, id);
        return Punto3D(x/magnitud, y/magnitud, z/magnitud, id);
    }
    
    float magnitud() const {
        return std::sqrt(x*x + y*y + z*z);
    }
};

struct ResultadoVecino {
    int id;
    float distancia;
    
    ResultadoVecino() : id(-1), distancia(std::numeric_limits<float>::max()) {}
    ResultadoVecino(int id_, float dist_) : id(id_), distancia(dist_) {}
    
    bool operator<(const ResultadoVecino& otro) const {
        return distancia < otro.distancia;
    }
};

// ============================================================================
// FUNCIONES DE DISTANCIA
// ============================================================================

inline float distancia_euclidiana(const Punto3D& a, const Punto3D& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

inline float distancia_manhattan(const Punto3D& a, const Punto3D& b) {
    return std::abs(a.x - b.x) + std::abs(a.y - b.y) + std::abs(a.z - b.z);
}

inline float distancia_chebyshev(const Punto3D& a, const Punto3D& b) {
    float dx = std::abs(a.x - b.x);
    float dy = std::abs(a.y - b.y);
    float dz = std::abs(a.z - b.z);
    return std::max({dx, dy, dz});
}

inline float distancia_coseno(const Punto3D& a, const Punto3D& b) {
    float producto_punto = a.x*b.x + a.y*b.y + a.z*b.z;
    return 1.0f - producto_punto;
}

// ============================================================================
// UTILIDADES DE TIEMPO
// ============================================================================

class Temporizador {
private:
    std::chrono::high_resolution_clock::time_point inicio;
    std::string nombre;
    bool silencioso;
    
public:
    Temporizador(const std::string& nombre_, bool silencioso_ = false) 
        : nombre(nombre_), silencioso(silencioso_) {
        inicio = std::chrono::high_resolution_clock::now();
    }
    
    ~Temporizador() {
        if (!silencioso) {
            auto fin = std::chrono::high_resolution_clock::now();
            auto duracion = std::chrono::duration_cast<std::chrono::milliseconds>(fin - inicio);
            std::cout << nombre << ": " << duracion.count() << " ms" << std::endl;
        }
    }
    
    double obtener_tiempo_ms() const {
        auto fin = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(fin - inicio).count();
    }
};

// ============================================================================
// CARGA DE DATOS CSV
// ============================================================================

class CargadorCSV {
public:
    static std::vector<Punto3D> cargar_datos(const std::string& ruta_archivo) {
        std::vector<Punto3D> puntos;
        std::ifstream archivo(ruta_archivo);
        
        if (!archivo.is_open()) {
            std::cerr << "Error: No se pudo abrir " << ruta_archivo << std::endl;
            return puntos;
        }
        
        std::string linea;
        std::getline(archivo, linea); // Saltar encabezado
        
        while (std::getline(archivo, linea)) {
            std::stringstream ss(linea);
            std::string valor;
            int id;
            float x, y, z;
            
            std::getline(ss, valor, ',');
            id = std::stoi(valor);
            
            std::getline(ss, valor, ',');
            x = std::stof(valor);
            
            std::getline(ss, valor, ',');
            y = std::stof(valor);
            
            std::getline(ss, valor, ',');
            z = std::stof(valor);
            
            puntos.emplace_back(x, y, z, id);
        }
        
        archivo.close();
        std::cout << "Cargados " << puntos.size() << " puntos desde " << ruta_archivo << std::endl;
        return puntos;
    }
    
    static std::vector<Punto3D> cargar_queries(const std::string& ruta_archivo) {
        std::vector<Punto3D> queries;
        std::ifstream archivo(ruta_archivo);
        
        if (!archivo.is_open()) {
            std::cerr << "Error: No se pudo abrir " << ruta_archivo << std::endl;
            return queries;
        }
        
        std::string linea;
        std::getline(archivo, linea); // Saltar encabezado
        
        int id = 0;
        while (std::getline(archivo, linea)) {
            std::stringstream ss(linea);
            std::string valor;
            float x, y, z;
            
            std::getline(ss, valor, ',');
            x = std::stof(valor);
            
            std::getline(ss, valor, ',');
            y = std::stof(valor);
            
            std::getline(ss, valor, ',');
            z = std::stof(valor);
            
            queries.emplace_back(x, y, z, id++);
        }
        
        archivo.close();
        std::cout << "Cargadas " << queries.size() << " queries desde " << ruta_archivo << std::endl;
        return queries;
    }
    
    struct ResultadosKNN {
        std::vector<std::vector<int>> vecinos_ids;
        std::vector<std::vector<float>> vecinos_dists;
    };
    
    static ResultadosKNN cargar_resultados_knn(const std::string& ruta_archivo, int k = 10) {
        ResultadosKNN resultados;
        std::ifstream archivo(ruta_archivo);
        
        if (!archivo.is_open()) {
            std::cerr << "Error: No se pudo abrir " << ruta_archivo << std::endl;
            return resultados;
        }
        
        std::string linea;
        std::getline(archivo, linea); // Saltar encabezado
        
        while (std::getline(archivo, linea)) {
            std::stringstream ss(linea);
            std::string valor;
            
            std::vector<int> ids(k);
            std::vector<float> dists(k);
            
            for (int i = 0; i < k; i++) {
                std::getline(ss, valor, ',');
                ids[i] = std::stoi(valor);
            }
            
            for (int i = 0; i < k; i++) {
                std::getline(ss, valor, ',');
                dists[i] = std::stof(valor);
            }
            
            resultados.vecinos_ids.push_back(ids);
            resultados.vecinos_dists.push_back(dists);
        }
        
        archivo.close();
        return resultados;
    }
};

// ============================================================================
// EXPORTADOR DE RESULTADOS
// ============================================================================

class ExportadorResultados {
public:
    static void guardar_resultados_knn(
        const std::string& ruta_archivo,
        const std::vector<std::vector<ResultadoVecino>>& resultados,
        int k = 10) {
        
        std::ofstream archivo(ruta_archivo);
        if (!archivo.is_open()) {
            std::cerr << "Error: No se pudo crear " << ruta_archivo << std::endl;
            return;
        }
        
        // Escribir encabezado
        for (int i = 0; i < k; i++) {
            archivo << "neighbor_" << i << "_id";
            if (i < k-1) archivo << ",";
        }
        for (int i = 0; i < k; i++) {
            archivo << ",neighbor_" << i << "_dist";
        }
        archivo << "\n";
        
        // Escribir resultados
        for (const auto& resultado_query : resultados) {
            for (int i = 0; i < k && i < resultado_query.size(); i++) {
                archivo << resultado_query[i].id;
                if (i < k-1) archivo << ",";
            }
            for (int i = 0; i < k && i < resultado_query.size(); i++) {
                archivo << "," << resultado_query[i].distancia;
            }
            archivo << "\n";
        }
        
        archivo.close();
        std::cout << "Resultados guardados en " << ruta_archivo << std::endl;
    }
};

#endif // UTILIDADES_H
