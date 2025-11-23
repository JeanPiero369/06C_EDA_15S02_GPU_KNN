// ============================================================================
// ARKADE GPU k-NN - Interfaz Gráfica con ImGui + OpenGL + GLFW
// ============================================================================

#define NOMINMAX
#include <Windows.h>
#include <GL/gl.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

// ============================================================================
// ESTRUCTURAS DE DATOS
// ============================================================================

struct Punto3D {
    float x, y, z;
    int id;
};

struct ResultadoKNN {
    int id;
    float distancia;
};

struct QueryResult {
    int query_idx;
    std::vector<ResultadoKNN> vecinos;
};

// ============================================================================
// ESTADO DE LA APLICACIÓN
// ============================================================================

struct AppState {
    // Archivos disponibles
    std::vector<std::string> archivos_data;
    std::vector<std::string> archivos_queries;
    std::vector<std::string> archivos_results;
    
    // Selecciones actuales
    int selected_data_idx = 0;
    int selected_queries_idx = 0;
    int selected_distance_idx = 0;
    int selected_method_idx = 0;
    int selected_query_row = 0;
    
    // Parámetros de búsqueda
    int k_vecinos = 10;
    float radio_busqueda = 100.0f;
    
    // Datos cargados
    std::vector<Punto3D> puntos;
    std::vector<Punto3D> queries;
    std::vector<QueryResult> resultados;
    
    // Estado de visualización
    bool datos_cargados = false;
    bool queries_cargados = false;
    bool resultados_disponibles = false;
    
    // Tiempo de búsqueda
    double tiempo_busqueda_ms = 0.0;
    std::string mensaje_estado = "Listo";
    
    // Parámetros de cámara para el viewport
    float camera_distance = 2000.0f;
    float camera_rot_x = 30.0f;
    float camera_rot_y = 45.0f;
    float point_size = 2.0f;
    
    // Centro de la cámara (para zoom en query)
    float camera_center_x = 0.0f;
    float camera_center_y = 0.0f;
    float camera_center_z = 0.0f;
    bool zoom_en_query = false;
    bool ocultar_no_encontrados = false;  // Por defecto mostrar todos los puntos
    
    // Opciones
    const char* distancias[4] = {"L2 (Euclidean)", "L1 (Manhattan)", "Linf (Chebyshev)", "Cosine"};
    const char* metodos[3] = {"ARKADE (GPU RT Cores)", "FAISS (CPU)", "FLANN (CPU)"};
    const char* dist_codes[4] = {"l2", "l1", "linf", "cosine"};
};

static AppState g_app;

// ============================================================================
// FUNCIONES DE CARGA DE ARCHIVOS
// ============================================================================

void escanear_archivos() {
    g_app.archivos_data.clear();
    g_app.archivos_queries.clear();
    g_app.archivos_results.clear();
    
    // Escanear carpeta data/
    if (fs::exists("data")) {
        for (const auto& entry : fs::directory_iterator("data")) {
            if (entry.path().extension() == ".csv") {
                std::string filename = entry.path().filename().string();
                // Archivos que empiezan con "data"
                if (filename.find("data") == 0) {
                    g_app.archivos_data.push_back(filename);
                }
                // Archivos que empiezan con "queries"
                if (filename.find("queries") == 0) {
                    g_app.archivos_queries.push_back(filename);
                }
            }
        }
    }
    
    // Escanear carpeta results/
    if (fs::exists("results")) {
        for (const auto& entry : fs::directory_iterator("results")) {
            if (entry.path().extension() == ".csv") {
                g_app.archivos_results.push_back(entry.path().filename().string());
            }
        }
    }
    
    std::sort(g_app.archivos_data.begin(), g_app.archivos_data.end());
    std::sort(g_app.archivos_queries.begin(), g_app.archivos_queries.end());
    std::sort(g_app.archivos_results.begin(), g_app.archivos_results.end());
}

bool cargar_puntos(const std::string& filename) {
    std::string path = "data/" + filename;
    std::ifstream file(path);
    if (!file.is_open()) {
        g_app.mensaje_estado = "Error: No se pudo abrir " + filename;
        return false;
    }
    
    g_app.puntos.clear();
    std::string line;
    bool header = true;
    int id = 0;
    
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }
        
        std::stringstream ss(line);
        std::string token;
        Punto3D p;
        
        // Formato: id,x,y,z o x,y,z
        std::vector<float> valores;
        while (std::getline(ss, token, ',')) {
            try {
                valores.push_back(std::stof(token));
            } catch (...) {}
        }
        
        if (valores.size() >= 4) {
            p.id = (int)valores[0];
            p.x = valores[1];
            p.y = valores[2];
            p.z = valores[3];
        } else if (valores.size() >= 3) {
            p.id = id++;
            p.x = valores[0];
            p.y = valores[1];
            p.z = valores[2];
        } else {
            continue;
        }
        
        g_app.puntos.push_back(p);
    }
    
    g_app.datos_cargados = true;
    g_app.mensaje_estado = "Cargados " + std::to_string(g_app.puntos.size()) + " puntos de " + filename;
    return true;
}

bool cargar_queries(const std::string& filename) {
    std::string path = "data/" + filename;
    std::ifstream file(path);
    if (!file.is_open()) {
        g_app.mensaje_estado = "Error: No se pudo abrir " + filename;
        return false;
    }
    
    g_app.queries.clear();
    std::string line;
    bool header = true;
    int id = 0;
    
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }
        
        std::stringstream ss(line);
        std::string token;
        Punto3D p;
        
        std::vector<float> valores;
        while (std::getline(ss, token, ',')) {
            try {
                valores.push_back(std::stof(token));
            } catch (...) {}
        }
        
        if (valores.size() >= 3) {
            p.id = id++;
            p.x = valores[0];
            p.y = valores[1];
            p.z = valores[2];
            g_app.queries.push_back(p);
        }
    }
    
    g_app.queries_cargados = true;
    g_app.mensaje_estado = "Cargados " + std::to_string(g_app.queries.size()) + " queries de " + filename;
    return true;
}

bool cargar_resultados(const std::string& filename) {
    std::string path = "results/" + filename;
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }
    
    g_app.resultados.clear();
    std::string line;
    bool header = true;
    int query_idx = 0;
    
    while (std::getline(file, line)) {
        if (header) { header = false; continue; }
        
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        // Formato: neighbor_0_id,...,neighbor_k_id,neighbor_0_dist,...,neighbor_k_dist
        int k = (int)tokens.size() / 2;
        QueryResult qr;
        qr.query_idx = query_idx++;
        
        for (int i = 0; i < k; i++) {
            ResultadoKNN r;
            try {
                r.id = std::stoi(tokens[i]);
                r.distancia = std::stof(tokens[k + i]);
                qr.vecinos.push_back(r);
            } catch (...) {}
        }
        
        g_app.resultados.push_back(qr);
    }
    
    g_app.resultados_disponibles = !g_app.resultados.empty();
    return g_app.resultados_disponibles;
}

// ============================================================================
// FUNCIÓN DE BÚSQUEDA (llama al ejecutable principal)
// ============================================================================

// Función para leer el tiempo de búsqueda del archivo generado por arkade_knn
double leer_tiempo_busqueda() {
    std::ifstream file("results/last_search_time.txt");
    if (file.is_open()) {
        double tiempo;
        file >> tiempo;
        file.close();
        return tiempo;
    }
    return -1.0;
}

void ejecutar_busqueda() {
    if (!g_app.datos_cargados) {
        g_app.mensaje_estado = "Error: Primero carga los datos";
        return;
    }
    
    // Construir comando
    std::string distancia = g_app.dist_codes[g_app.selected_distance_idx];
    std::string cmd = "build\\bin\\arkade_knn.exe " + 
                      std::to_string(g_app.radio_busqueda) + " " +
                      std::to_string(g_app.k_vecinos) + " " +
                      distancia;
    
    g_app.mensaje_estado = "Ejecutando búsqueda...";
    
    // Ejecutar comando
    int result = system(cmd.c_str());
    
    if (result == 0) {
        // Leer tiempo de búsqueda real (solo k-NN, sin carga de datos)
        g_app.tiempo_busqueda_ms = leer_tiempo_busqueda();
        
        // Cargar resultados según la distancia seleccionada
        std::string result_file;
        switch (g_app.selected_distance_idx) {
            case 0: result_file = "ARKADE_knn_euclidean.csv"; break;
            case 1: result_file = "ARKADE_knn_manhattan.csv"; break;
            case 2: result_file = "ARKADE_knn_chebyshev.csv"; break;
            case 3: result_file = "ARKADE_knn_cosine.csv"; break;
        }
        
        if (cargar_resultados(result_file)) {
            char buffer[64];
            snprintf(buffer, sizeof(buffer), "%.2f", g_app.tiempo_busqueda_ms);
            g_app.mensaje_estado = "Busqueda k-NN completada en " + std::string(buffer) + " ms";
            
            // Hacer zoom automático al primer query y sus vecinos
            if (!g_app.queries.empty() && !g_app.resultados.empty()) {
                g_app.selected_query_row = 0;
                const auto& q = g_app.queries[0];
                g_app.camera_center_x = q.x;
                g_app.camera_center_y = q.y;
                g_app.camera_center_z = q.z;
                g_app.zoom_en_query = true;
                
                // Calcular distancia máxima del vecino más lejano para ajustar zoom
                float max_dist = 0.0f;
                for (const auto& v : g_app.resultados[0].vecinos) {
                    if (v.distancia > max_dist) max_dist = v.distancia;
                }
                g_app.camera_distance = std::max(100.0f, max_dist * 3.0f);
            }
        } else {
            g_app.mensaje_estado = "Búsqueda completada pero no se encontró archivo de resultados";
        }
        
        // Actualizar lista de archivos
        escanear_archivos();
    } else {
        g_app.mensaje_estado = "Error en la búsqueda (código: " + std::to_string(result) + ")";
    }
}

// ============================================================================
// RENDERIZADO DEL VIEWPORT 3D
// ============================================================================

void renderizar_viewport(int width, int height) {
    glViewport(0, 0, width, height);
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (!g_app.datos_cargados || g_app.puntos.empty()) return;
    
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
    
    // Configurar proyección
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    float aspect = (float)width / (float)height;
    float fov = 45.0f;
    float near_plane = 1.0f;
    float far_plane = 10000.0f;
    float top = near_plane * tanf(fov * 3.14159f / 360.0f);
    float right = top * aspect;
    glFrustum(-right, right, -top, top, near_plane, far_plane);
    
    // Configurar vista
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0, 0, -g_app.camera_distance);
    glRotatef(g_app.camera_rot_x, 1, 0, 0);
    glRotatef(g_app.camera_rot_y, 0, 1, 0);
    // Trasladar al centro de la cámara (query seleccionado si hay zoom)
    glTranslatef(-g_app.camera_center_x, -g_app.camera_center_y, -g_app.camera_center_z);
    
    // Dibujar ejes (relativos al centro)
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // X - Rojo
    glColor3f(1, 0, 0); glVertex3f(g_app.camera_center_x, g_app.camera_center_y, g_app.camera_center_z); 
    glVertex3f(g_app.camera_center_x + 500, g_app.camera_center_y, g_app.camera_center_z);
    // Y - Verde
    glColor3f(0, 1, 0); glVertex3f(g_app.camera_center_x, g_app.camera_center_y, g_app.camera_center_z); 
    glVertex3f(g_app.camera_center_x, g_app.camera_center_y + 500, g_app.camera_center_z);
    // Z - Azul
    glColor3f(0, 0, 1); glVertex3f(g_app.camera_center_x, g_app.camera_center_y, g_app.camera_center_z); 
    glVertex3f(g_app.camera_center_x, g_app.camera_center_y, g_app.camera_center_z + 500);
    glEnd();
    
    // Crear set de IDs encontrados si hay resultados y se quiere ocultar
    std::vector<bool> es_vecino(g_app.puntos.size(), false);
    if (g_app.ocultar_no_encontrados && g_app.resultados_disponibles && 
        g_app.selected_query_row < (int)g_app.resultados.size()) {
        const auto& qr = g_app.resultados[g_app.selected_query_row];
        for (const auto& v : qr.vecinos) {
            if (v.id >= 0 && v.id < (int)g_app.puntos.size()) {
                es_vecino[v.id] = true;
            }
        }
    }
    
    // Dibujar puntos de datos (muestreo para rendimiento)
    glPointSize(g_app.point_size);
    glBegin(GL_POINTS);
    glColor3f(0.3f, 0.6f, 1.0f);  // Azul claro
    
    int step = std::max(1, (int)g_app.puntos.size() / 50000);
    for (size_t i = 0; i < g_app.puntos.size(); i += step) {
        // Si ocultar_no_encontrados está activo y hay resultados, solo mostrar vecinos
        if (g_app.ocultar_no_encontrados && g_app.resultados_disponibles) {
            if (!es_vecino[i]) continue;  // Saltar puntos que no son vecinos
        }
        const auto& p = g_app.puntos[i];
        glVertex3f(p.x, p.y, p.z);
    }
    glEnd();
    
    // Dibujar queries (si están cargados)
    if (g_app.queries_cargados && !g_app.queries.empty()) {
        glPointSize(g_app.point_size * 3);
        glBegin(GL_POINTS);
        glColor3f(1.0f, 0.3f, 0.3f);  // Rojo
        
        for (const auto& q : g_app.queries) {
            glVertex3f(q.x, q.y, q.z);
        }
        glEnd();
    }
    
    // Resaltar query seleccionado y sus vecinos
    if (g_app.resultados_disponibles && g_app.selected_query_row < (int)g_app.resultados.size()) {
        const auto& qr = g_app.resultados[g_app.selected_query_row];
        
        if (g_app.selected_query_row < (int)g_app.queries.size()) {
            const auto& q = g_app.queries[g_app.selected_query_row];
            glPointSize(g_app.point_size * 5);
            glBegin(GL_POINTS);
            glColor3f(1.0f, 1.0f, 0.0f);
            glVertex3f(q.x, q.y, q.z);
            glEnd();
            
            // Vecinos encontrados - verde
            glPointSize(g_app.point_size * 4);
            glBegin(GL_POINTS);
            glColor3f(0.0f, 1.0f, 0.0f);
            for (const auto& v : qr.vecinos) {
                if (v.id >= 0 && v.id < (int)g_app.puntos.size()) {
                    const auto& p = g_app.puntos[v.id];
                    glVertex3f(p.x, p.y, p.z);
                }
            }
            glEnd();
            
            // Líneas desde query a vecinos
            glLineWidth(1.0f);
            glBegin(GL_LINES);
            glColor4f(0.0f, 1.0f, 0.0f, 0.5f);
            for (const auto& v : qr.vecinos) {
                if (v.id >= 0 && v.id < (int)g_app.puntos.size()) {
                    const auto& p = g_app.puntos[v.id];
                    glVertex3f(q.x, q.y, q.z);
                    glVertex3f(p.x, p.y, p.z);
                }
            }
            glEnd();
        }
    }
    
    glDisable(GL_DEPTH_TEST);
}

// ============================================================================
// INTERFAZ DE USUARIO CON IMGUI
// ============================================================================

void renderizar_ui(int window_width, int window_height) {
    // Calcular dimensiones del panel derecho
    float panel_width = 380.0f;
    float control_panel_height = (float)(window_height - 20) * 0.55f;  // 55% superior
    float results_panel_height = (float)(window_height - 20) * 0.45f;  // 45% inferior
    float panel_x = (float)window_width - panel_width - 10;  // Paneles a la derecha
    
    // === PANEL SUPERIOR DERECHO: CONTROL ===
    ImGui::SetNextWindowPos(ImVec2(panel_x, 10), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panel_width, control_panel_height), ImGuiCond_Always);
    
    ImGui::Begin("ARKADE k-NN Control Panel", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    
    // === SECCIÓN: CARGA DE DATOS ===
    if (ImGui::CollapsingHeader("Carga de Datos", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Selector de archivo de datos
        ImGui::Text("Archivo de Puntos (data*):");
        if (!g_app.archivos_data.empty()) {
            std::vector<const char*> items;
            for (const auto& f : g_app.archivos_data) items.push_back(f.c_str());
            
            if (ImGui::Combo("##data_file", &g_app.selected_data_idx, items.data(), (int)items.size())) {
                cargar_puntos(g_app.archivos_data[g_app.selected_data_idx]);
            }
        } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No hay archivos data*.csv");
        }
        
        if (ImGui::Button("Cargar Puntos") && !g_app.archivos_data.empty()) {
            cargar_puntos(g_app.archivos_data[g_app.selected_data_idx]);
        }
        
        ImGui::Separator();
        
        // Selector de archivo de queries
        ImGui::Text("Archivo de Queries (queries*):");
        if (!g_app.archivos_queries.empty()) {
            std::vector<const char*> items;
            for (const auto& f : g_app.archivos_queries) items.push_back(f.c_str());
            
            if (ImGui::Combo("##queries_file", &g_app.selected_queries_idx, items.data(), (int)items.size())) {
                cargar_queries(g_app.archivos_queries[g_app.selected_queries_idx]);
            }
        } else {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1), "No hay archivos queries*.csv");
        }
        
        if (ImGui::Button("Cargar Queries") && !g_app.archivos_queries.empty()) {
            cargar_queries(g_app.archivos_queries[g_app.selected_queries_idx]);
        }
        
        ImGui::Separator();
        
        if (ImGui::Button("Refrescar Archivos")) {
            escanear_archivos();
        }
    }
    
    ImGui::Spacing();
    
    // === SECCIÓN: CONFIGURACIÓN DE BÚSQUEDA ===
    if (ImGui::CollapsingHeader("Configuracion de Busqueda", ImGuiTreeNodeFlags_DefaultOpen)) {
        // Tipo de distancia
        ImGui::Text("Tipo de Distancia:");
        ImGui::Combo("##distancia", &g_app.selected_distance_idx, g_app.distancias, 4);
        
        // Método/Librería
        ImGui::Text("Metodo/Libreria:");
        ImGui::Combo("##metodo", &g_app.selected_method_idx, g_app.metodos, 3);
        
        ImGui::Separator();
        
        // Parámetros
        ImGui::Text("Cantidad de Vecinos (K):");
        ImGui::InputInt("##k", &g_app.k_vecinos);
        g_app.k_vecinos = std::max(1, std::min(100, g_app.k_vecinos));
        
        ImGui::Text("Radio de Busqueda:");
        ImGui::InputFloat("##radio", &g_app.radio_busqueda, 10.0f, 100.0f, "%.1f");
        g_app.radio_busqueda = std::max(1.0f, g_app.radio_busqueda);
        
        ImGui::Separator();
        
        // Botón de búsqueda
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.3f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.3f, 0.8f, 0.4f, 1.0f));
        if (ImGui::Button("EJECUTAR BUSQUEDA", ImVec2(-1, 40))) {
            ejecutar_busqueda();
        }
        ImGui::PopStyleColor(2);
        
        // Mostrar tiempo
        if (g_app.tiempo_busqueda_ms > 0) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Tiempo: %.2f ms", g_app.tiempo_busqueda_ms);
        }
    }
    
    ImGui::Spacing();
    
    // === SECCIÓN: VISUALIZACIÓN ===
    if (ImGui::CollapsingHeader("Visualizacion 3D", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Distancia Camara", &g_app.camera_distance, 100.0f, 5000.0f);
        ImGui::SliderFloat("Rotacion X", &g_app.camera_rot_x, -90.0f, 90.0f);
        ImGui::SliderFloat("Rotacion Y", &g_app.camera_rot_y, -180.0f, 180.0f);
        ImGui::SliderFloat("Tamano Puntos", &g_app.point_size, 1.0f, 10.0f);
        
        ImGui::Separator();
        
        // Controles de zoom
        if (g_app.zoom_en_query) {
            ImGui::TextColored(ImVec4(0, 1, 0.5f, 1), "Zoom activo en Query %d", g_app.selected_query_row);
            if (ImGui::Button("Mostrar Todos los Puntos", ImVec2(-1, 0))) {
                g_app.zoom_en_query = false;
                g_app.camera_center_x = 0.0f;
                g_app.camera_center_y = 0.0f;
                g_app.camera_center_z = 0.0f;
                g_app.camera_distance = 2000.0f;
            }
        } else {
            if (g_app.resultados_disponibles && ImGui::Button("Zoom en Query Seleccionado", ImVec2(-1, 0))) {
                if (g_app.selected_query_row < (int)g_app.queries.size()) {
                    const auto& q = g_app.queries[g_app.selected_query_row];
                    g_app.camera_center_x = q.x;
                    g_app.camera_center_y = q.y;
                    g_app.camera_center_z = q.z;
                    g_app.zoom_en_query = true;
                    
                    // Ajustar zoom según distancia del vecino más lejano
                    if (g_app.selected_query_row < (int)g_app.resultados.size()) {
                        float max_dist = 0.0f;
                        for (const auto& v : g_app.resultados[g_app.selected_query_row].vecinos) {
                            if (v.distancia > max_dist) max_dist = v.distancia;
                        }
                        g_app.camera_distance = std::max(100.0f, max_dist * 3.0f);
                    }
                }
            }
        }
        
        // Info
        ImGui::Separator();
        ImGui::Text("Puntos cargados: %d", (int)g_app.puntos.size());
        ImGui::Text("Queries cargados: %d", (int)g_app.queries.size());
    }
    
    ImGui::Spacing();
    
    // === SECCIÓN: ESTADO ===
    ImGui::Separator();
    ImGui::TextWrapped("Estado: %s", g_app.mensaje_estado.c_str());
    
    ImGui::End();
    
    // === PANEL INFERIOR DERECHO: RESULTADOS ===
    ImGui::SetNextWindowPos(ImVec2(panel_x, 10 + control_panel_height + 5), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(panel_width, results_panel_height - 15), ImGuiCond_Always);
    
    ImGui::Begin("Resultados de Busqueda", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
    
    if (g_app.resultados_disponibles && !g_app.resultados.empty()) {
        // Selector de query
        ImGui::Text("Query seleccionado:");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        int prev_query = g_app.selected_query_row;
        ImGui::InputInt("##query_sel", &g_app.selected_query_row);
        g_app.selected_query_row = std::max(0, std::min((int)g_app.resultados.size() - 1, g_app.selected_query_row));
        
        ImGui::SameLine();
        ImGui::Text("/%d", (int)g_app.resultados.size() - 1);
        
        // Botones de navegación
        if (ImGui::Button("<", ImVec2(30, 0))) {
            g_app.selected_query_row = std::max(0, g_app.selected_query_row - 1);
        }
        ImGui::SameLine();
        if (ImGui::Button(">", ImVec2(30, 0))) {
            g_app.selected_query_row = std::min((int)g_app.resultados.size() - 1, g_app.selected_query_row + 1);
        }
        ImGui::SameLine();
        if (ImGui::Button("Zoom", ImVec2(50, 0))) {
            if (g_app.selected_query_row < (int)g_app.queries.size()) {
                const auto& q = g_app.queries[g_app.selected_query_row];
                g_app.camera_center_x = q.x;
                g_app.camera_center_y = q.y;
                g_app.camera_center_z = q.z;
                g_app.zoom_en_query = true;
                
                float max_dist = 0.0f;
                for (const auto& v : g_app.resultados[g_app.selected_query_row].vecinos) {
                    if (v.distancia > max_dist) max_dist = v.distancia;
                }
                g_app.camera_distance = std::max(100.0f, max_dist * 3.0f);
            }
        }
        
        // Si el query cambió y hay zoom activo, actualizar centro
        if (prev_query != g_app.selected_query_row && g_app.zoom_en_query) {
            if (g_app.selected_query_row < (int)g_app.queries.size()) {
                const auto& q = g_app.queries[g_app.selected_query_row];
                g_app.camera_center_x = q.x;
                g_app.camera_center_y = q.y;
                g_app.camera_center_z = q.z;
                
                float max_dist = 0.0f;
                for (const auto& v : g_app.resultados[g_app.selected_query_row].vecinos) {
                    if (v.distancia > max_dist) max_dist = v.distancia;
                }
                g_app.camera_distance = std::max(100.0f, max_dist * 3.0f);
            }
        }
        
        // Mostrar coordenadas del query
        if (g_app.selected_query_row < (int)g_app.queries.size()) {
            const auto& q = g_app.queries[g_app.selected_query_row];
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Query: (%.1f, %.1f, %.1f)", q.x, q.y, q.z);
        }
        
        // Checkbox para ocultar puntos no encontrados
        ImGui::Checkbox("Ocultar puntos no retornados", &g_app.ocultar_no_encontrados);
        
        ImGui::Separator();
        
        // Tabla de resultados
        if (ImGui::BeginTable("ResultsTable", 3, 
            ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY,
            ImVec2(0, 0))) {  // Usar todo el espacio disponible
            
            ImGui::TableSetupColumn("Rank", ImGuiTableColumnFlags_WidthFixed, 50);
            ImGui::TableSetupColumn("ID Vecino", ImGuiTableColumnFlags_WidthFixed, 100);
            ImGui::TableSetupColumn("Distancia", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableHeadersRow();
            
            const auto& qr = g_app.resultados[g_app.selected_query_row];
            for (int i = 0; i < (int)qr.vecinos.size(); i++) {
                ImGui::TableNextRow();
                
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%d", i + 1);
                
                ImGui::TableSetColumnIndex(1);
                ImGui::Text("%d", qr.vecinos[i].id);
                
                ImGui::TableSetColumnIndex(2);
                ImGui::Text("%.6f", qr.vecinos[i].distancia);
            }
            
            ImGui::EndTable();
        }
    } else {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "No hay resultados disponibles.");
        ImGui::Text("Ejecuta una busqueda para ver los resultados.");
        
        // Cargar resultados existentes
        if (!g_app.archivos_results.empty()) {
            ImGui::Separator();
            ImGui::Text("Cargar resultados existentes:");
            
            static int selected_result = 0;
            std::vector<const char*> items;
            for (const auto& f : g_app.archivos_results) items.push_back(f.c_str());
            
            ImGui::Combo("##result_file", &selected_result, items.data(), (int)items.size());
            ImGui::SameLine();
            if (ImGui::Button("Cargar")) {
                if (cargar_resultados(g_app.archivos_results[selected_result])) {
                    g_app.mensaje_estado = "Resultados cargados: " + g_app.archivos_results[selected_result];
                }
            }
        }
    }
    
    ImGui::End();
}

// ============================================================================
// CALLBACKS DE GLFW
// ============================================================================

void glfw_error_callback(int error, const char* description) {
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    // Inicializar GLFW
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        std::cerr << "Error: No se pudo inicializar GLFW" << std::endl;
        return -1;
    }
    
    // Configuración de OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    
    // Crear ventana
    GLFWwindow* window = glfwCreateWindow(1400, 900, "ARKADE GPU k-NN Visualizer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Error: No se pudo crear la ventana" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // VSync
    
    // Inicializar ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Estilo oscuro
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    
    // Inicializar backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 120");
    
    // Escanear archivos disponibles
    escanear_archivos();
    
    // Cargar datos iniciales si existen
    if (!g_app.archivos_data.empty()) {
        cargar_puntos(g_app.archivos_data[0]);
    }
    if (!g_app.archivos_queries.empty()) {
        cargar_queries(g_app.archivos_queries[0]);
    }
    
    // Loop principal
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Obtener dimensiones de la ventana
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        
        // Nueva frame de ImGui
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        
        // Control de cámara con mouse (fuera de ventanas ImGui)
        if (!io.WantCaptureMouse) {
            if (io.MouseDown[0]) {  // Click izquierdo - rotar
                g_app.camera_rot_y += io.MouseDelta.x * 0.5f;
                g_app.camera_rot_x += io.MouseDelta.y * 0.5f;
                g_app.camera_rot_x = std::max(-90.0f, std::min(90.0f, g_app.camera_rot_x));
            }
            if (io.MouseWheel != 0) {  // Scroll - zoom
                g_app.camera_distance -= io.MouseWheel * 100.0f;
                g_app.camera_distance = std::max(100.0f, std::min(5000.0f, g_app.camera_distance));
            }
        }
        
        // Renderizar UI pasando dimensiones
        renderizar_ui(display_w, display_h);
        
        // Calcular área del viewport 3D (lado izquierdo)
        int panel_width = 400;  // Ancho del panel derecho
        int viewport_x = 0;     // Viewport empieza desde la izquierda
        int viewport_w = display_w - panel_width;
        int viewport_h = display_h;
        
        // Limpiar el fondo primero
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Renderizar viewport 3D en el área izquierda
        glViewport(viewport_x, 0, viewport_w, viewport_h);
        renderizar_viewport(viewport_w, viewport_h);
        
        // Reset viewport para ImGui
        glViewport(0, 0, display_w, display_h);
        
        // Renderizar ImGui
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        glfwSwapBuffers(window);
    }
    
    // Limpieza
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    return 0;
}
