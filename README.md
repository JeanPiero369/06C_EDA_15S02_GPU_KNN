# Arkade: k-NN Search con GPU Ray Tracing

Implementación completa del paper **"Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing"** en C++ usando NVIDIA OptiX.

## 📋 Descripción

Este proyecto implementa búsqueda k-NN para 4 tipos de distancias usando aceleración por GPU:

- **L2 (Euclidean)**: Geometría esférica (52% AABB occupancy)
- **L1 (Manhattan)**: Geometría bipiramidal (33% AABB occupancy)
- **L∞ (Chebyshev)**: Geometría cúbica (100% AABB occupancy - sin fase refine)
- **Cosine**: Normalización + L2 con transformación monotónica

### Metodología Filter-Refine

1. **FILTER**: RT cores hacen intersección rayo-AABB (BVH traversal hardware)
2. **REFINE**: Shader cores calculan distancia exacta dentro de AABB

## 🔧 Requisitos

### Hardware
- GPU NVIDIA con RT cores (RTX 2000+ series)
- Mínimo 4GB VRAM

### Software
- **NVIDIA OptiX SDK 7.5+** (ya instalado)
- **CUDA Toolkit 11.0+** (ya instalado)
- **CMake 3.18+**
- **Visual Studio 2019+** (en Windows)
- **FAISS** (Facebook AI Similarity Search)
- **NMSLIB** (Non-Metric Space Library con HNSW)
- **FLANN** (Fast Library for Approximate Nearest Neighbors)

## 📦 Instalación de Dependencias

### FAISS (CPU + GPU)

#### Windows con vcpkg:
```powershell
# Instalar vcpkg si no lo tienes
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Instalar FAISS
.\vcpkg install faiss:x64-windows
.\vcpkg install faiss[gpu]:x64-windows
```

#### Desde fuente:
```powershell
git clone https://github.com/facebookresearch/faiss.git
cd faiss
cmake -B build -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF
cmake --build build --config Release
cmake --install build --prefix "C:/Program Files/faiss"
```

### NMSLIB

```powershell
git clone https://github.com/nmslib/nmslib.git
cd nmslib/similarity_search
cmake -B build
cmake --build build --config Release
cmake --install build --prefix "C:/Program Files/nmslib"
```

### FLANN

#### Windows con vcpkg:
```powershell
.\vcpkg install flann:x64-windows
```

#### Desde fuente:
```powershell
git clone https://github.com/flann-lib/flann.git
cd flann
cmake -B build -DBUILD_PYTHON_BINDINGS=OFF -DBUILD_MATLAB_BINDINGS=OFF
cmake --build build --config Release
cmake --install build --prefix "C:/Program Files/flann"
```

**Nota**: FLANN es una librería CPU optimizada para búsqueda de vecinos cercanos usando KD-Trees.

## 🏗️ Compilación

```powershell
# Navegar al directorio del proyecto
cd "E:\06. Sexto Ciclo\02. Advanced Data Structures\07. Workspace\15S01. Proyecto\06C_EDA_15S02_GPU_KNN"

# Crear directorio build
mkdir build
cd build

# Configurar con CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compilar
cmake --build . --config Release

# El ejecutable estará en: build/bin/arkade_knn.exe
```

### Ajustar paths en CMakeLists.txt

Si las librerías están en ubicaciones diferentes, editar:

```cmake
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
set(FAISS_ROOT "C:/Program Files/faiss")
set(NMSLIB_ROOT "C:/Program Files/nmslib")
set(FLANN_ROOT "C:/Program Files/flann")
```

## 🚀 Ejecución

```powershell
# Desde el directorio build
cd bin
.\arkade_knn.exe

# O desde el directorio raíz
.\build\bin\arkade_knn.exe
```

### Estructura de Datos

El programa espera los siguientes archivos en `data/`:

- `data.csv`: Dataset principal (1M puntos, formato: id,x,y,z)
- `queries.csv`: Consultas k-NN (10K queries, formato: x,y,z)
- `knn_euclidean.csv`: Ground truth para L2
- `knn_manhattan.csv`: Ground truth para L1
- `knn_chebyshev.csv`: Ground truth para L∞
- `knn_cosine.csv`: Ground truth para cosine

## 📊 Resultados

Los resultados se guardan en `results/` con el formato:

```
ARKADE_knn_euclidean.csv
ARKADE_knn_manhattan.csv
ARKADE_knn_chebyshev.csv
ARKADE_knn_cosine.csv
FAISS_CPU_knn_euclidean.csv
FAISS_GPU_knn_euclidean.csv
FAISS_CPU_knn_cosine.csv
FAISS_GPU_knn_cosine.csv
NMSLIB_knn_manhattan.csv
NMSLIB_knn_chebyshev.csv
FLANN_knn_manhattan.csv
FLANN_knn_chebyshev.csv
```

### Métricas de Validación

Para cada método y distancia, se calculan:

- **Precision**: Porcentaje de vecinos correctos encontrados
- **Recall**: Porcentaje de vecinos ground truth recuperados
- **Error de Distancia**: Diferencia promedio con ground truth
- **Exactitud**: Porcentaje de queries con recall ≥ 0.9

### Comparación de Métodos

El programa imprime tablas comparativas:

```
========================================
COMPARACIÓN: L2 (Euclidean)
========================================
Metodo              Precision      Recall   Error Dist   Exactitud (%)
--------------------------------------------------------------------------
Arkade OptiX           0.9980      0.9980     0.000123          99.50
FAISS CPU              1.0000      1.0000     0.000000         100.00
FAISS GPU              1.0000      1.0000     0.000000         100.00
==========================================================================
```

## 🧪 Arquitectura del Código

```
06C_EDA_15S02_GPU_KNN/
├── include/
│   ├── utilidades.h          # Estructuras base, CSV I/O, timer
│   ├── arkade_optix.h        # Clase principal Arkade con OptiX
│   ├── baseline_faiss.h      # FAISS CPU/GPU (L2, Cosine)
│   ├── baseline_nmslib.h     # NMSLIB HNSW (L1, L∞)
│   └── baseline_flann.h      # FLANN CPU KD-Tree (L1, L∞)
├── kernels/
│   └── arkade_kernels.cu     # Kernels OptiX/CUDA
├── src/
│   └── main.cpp              # Programa principal
├── data/                     # Datasets y ground truth
├── results/                  # Resultados de experimentos
├── build/                    # Directorio de compilación
└── CMakeLists.txt            # Configuración CMake
```

## 🔬 Detalles de Implementación

### Arkade OptiX (`arkade_optix.h`)

- **Inicialización**: Crea contexto OptiX con logging
- **GAS Building**: Construye Geometry Acceleration Structure con AABBs
- **Radius Query**: Búsqueda por radio usando ray tracing
- **k-NN Search**: Expansión adaptativa de radio hasta encontrar k vecinos

### Kernels CUDA (`arkade_kernels.cu`)

- `__raygen__rg`: Genera rayos desde puntos de consulta
- `__intersection__is`: Tests geométricos (esfera/bipiramide/cubo)
- `__closesthit__ch`: Registra hits y calcula distancias
- `busqueda_lineal_kernel`: Fallback linear search

### FAISS Baselines (`baseline_faiss.h`)

- **CPU**: `IndexFlatL2` (L2), `IndexFlat` con `METRIC_INNER_PRODUCT` (Cosine)
- **GPU**: `GpuIndexFlat` con `StandardGpuResources`
- Batch processing optimizado

### NMSLIB HNSW (`baseline_nmslib.h`)

- Espacios métricos: `l1` (Manhattan), `linf` (Chebyshev)
- HNSW con `M=16`, `efConstruction=200`
- Búsqueda secuencial de queries

### FLANN CPU (`baseline_flann.h`)

- Librería oficial: https://github.com/flann-lib/flann
- KD-Tree construido en CPU altamente optimizado
- Soporta métricas L1 (Manhattan) y L∞ (Chebyshev)
- Búsqueda aproximada rápida con parámetros ajustables

## 📈 Optimizaciones

1. **OptiX RT Cores**: Hardware acceleration para BVH traversal
2. **FAISS GPU**: Búsqueda vectorial masivamente paralela
3. **HNSW Graphs**: Navegación logarítmica en grafos
4. **KD-Tree GPU**: Particionamiento espacial paralelo
5. **Batch Processing**: Todas las queries en un solo kernel launch

## 🐛 Troubleshooting

### Error: OptiX SDK no encontrado

Verificar path en CMakeLists.txt:
```cmake
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0")
```

### Error: CUDA out of memory

Reducir tamaño de batch en `buscar_knn_batch`:
```cpp
const int BATCH_SIZE = 1000; // Reducir si hay errores de memoria
```

### Error: PTX compilation failed

Verificar arquitectura CUDA en CMakeLists.txt:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)  # sm_86 para RTX 3050 Ti
```

Para otras GPUs:
- RTX 4090: `89`
- RTX 3090: `86`
- RTX 2080: `75`

### Error: FAISS not found

Instalar con vcpkg o compilar desde fuente (ver sección de instalación).

## 📚 Referencias

- **Paper**: Arkade: k-Nearest Neighbor Search With Non-Euclidean Distances using GPU Ray Tracing
- **OptiX**: https://developer.nvidia.com/optix
- **FAISS**: https://github.com/facebookresearch/faiss
- **NMSLIB**: https://github.com/nmslib/nmslib
- **FLANN**: https://github.com/flann-lib/flann

## 👥 Autor

Proyecto de Advanced Data Structures - Sexto Ciclo

## 📄 Licencia

Académico - Universidad Nacional de Ingeniería
