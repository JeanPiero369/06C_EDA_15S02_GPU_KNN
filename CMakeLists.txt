cmake_minimum_required(VERSION 3.18)
project(Arkade_GPU_KNN LANGUAGES CXX CUDA)

# ============================================================================
# CONFIGURACIÓN GENERAL
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configuración de arquitectura CUDA (RTX 3050 Ti = sm_86)
set(CMAKE_CUDA_ARCHITECTURES 86)

# ============================================================================
# NVIDIA OptiX SDK
# ============================================================================

# Buscar OptiX SDK (ajustar path si necesario)
set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" CACHE PATH "Path to OptiX SDK")

if(NOT EXISTS ${OptiX_INSTALL_DIR})
    message(FATAL_ERROR "OptiX SDK no encontrado en: ${OptiX_INSTALL_DIR}")
endif()

set(OptiX_INCLUDE_DIR ${OptiX_INSTALL_DIR}/include)
include_directories(${OptiX_INCLUDE_DIR})

# ============================================================================
# CUDA
# ============================================================================

find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ============================================================================
# FAISS (Facebook AI Similarity Search)
# ============================================================================

include(CMakeFindDependencyMacro)
find_package(faiss CONFIG REQUIRED)
set(FAISS_LIBRARIES faiss)
message(STATUS "FAISS encontrado via vcpkg")

# ============================================================================
# NMSLIB (Non-Metric Space Library)
# ============================================================================

set(NMSLIB_ROOT "${CMAKE_SOURCE_DIR}/external/nmslib" CACHE PATH "Path to NMSLIB installation")

if(EXISTS ${NMSLIB_ROOT})
    include_directories(${NMSLIB_ROOT}/include)
    link_directories(${NMSLIB_ROOT}/lib)
    set(NMSLIB_LIBRARIES NonMetricSpaceLib)
    message(STATUS "NMSLIB encontrado: ${NMSLIB_ROOT}")
else()
    message(WARNING "NMSLIB no encontrado - compilando sin soporte NMSLIB")
    set(NMSLIB_LIBRARIES "")
endif()

# ============================================================================
# FLANN (Fast Library for Approximate Nearest Neighbors)
# ============================================================================

find_package(flann CONFIG REQUIRED)
set(FLANN_LIBRARIES flann::flann flann::flann_cpp)
message(STATUS "FLANN encontrado via vcpkg")

# ============================================================================
# DIRECTORIOS DEL PROYECTO
# ============================================================================

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/kernels)

# ============================================================================
# COMPILACIÓN DE KERNELS OptiX A PTX
# ============================================================================

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)

# Archivo de kernels OptiX
set(ARKADE_KERNELS ${PROJECT_SOURCE_DIR}/kernels/arkade_kernels.cu)

# Directorio de salida para PTX
set(PTX_OUTPUT_DIR ${CMAKE_BINARY_DIR}/ptx)
file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

# Custom command para compilar .cu a .ptx
add_custom_command(
    OUTPUT ${PTX_OUTPUT_DIR}/arkade_kernels.ptx
    COMMAND ${CMAKE_CUDA_COMPILER} --ptx -arch=sm_86 -use_fast_math 
        "-I${OptiX_INCLUDE_DIR}" 
        "-I${CUDAToolkit_INCLUDE_DIRS}" 
        "-I${PROJECT_SOURCE_DIR}/include" 
        "${ARKADE_KERNELS}" 
        -o "${PTX_OUTPUT_DIR}/arkade_kernels.ptx"
    DEPENDS ${ARKADE_KERNELS}
    COMMENT "Compilando kernels OptiX a PTX..."
    VERBATIM
)

add_custom_target(compile_ptx ALL
    DEPENDS ${PTX_OUTPUT_DIR}/arkade_kernels.ptx
)

# ============================================================================
# EJECUTABLE PRINCIPAL
# ============================================================================

add_executable(arkade_knn
    src/main.cpp
)

# Asegurar que PTX se compile antes del ejecutable
add_dependencies(arkade_knn compile_ptx)

# ============================================================================
# LINKEO DE LIBRERÍAS
# ============================================================================

target_link_libraries(arkade_knn
    CUDA::cudart
    CUDA::cuda_driver
    ${FAISS_LIBRARIES}
    ${FLANN_LIBRARIES}
    ${NMSLIB_LIBRARIES}
)

# OptiX es header-only excepto la inicialización
# Buscar optix_stubs
find_library(OPTIX_STUBS_LIB
    NAMES optix.7.6.0 optix.7.7.0 optix.8.0.0
    PATHS ${OptiX_INSTALL_DIR}/lib64
)

if(OPTIX_STUBS_LIB)
    target_link_libraries(arkade_knn ${OPTIX_STUBS_LIB})
    message(STATUS "OptiX stubs library: ${OPTIX_STUBS_LIB}")
endif()

# ============================================================================
# DEFINICIONES DE PREPROCESADOR
# ============================================================================

# Path al archivo PTX (para que el ejecutable lo encuentre)
target_compile_definitions(arkade_knn PRIVATE
    PTX_PATH="${PTX_OUTPUT_DIR}/arkade_kernels.ptx"
)

# ============================================================================
# COPIAR ARCHIVOS PTX AL DIRECTORIO DEL EJECUTABLE
# ============================================================================

add_custom_command(TARGET arkade_knn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${PTX_OUTPUT_DIR}/arkade_kernels.ptx
        $<TARGET_FILE_DIR:arkade_knn>/arkade_kernels.ptx
    COMMENT "Copiando PTX al directorio del ejecutable"
)

# ============================================================================
# CONFIGURACIÓN DE COMPILACIÓN
# ============================================================================

if(MSVC)
    # Visual Studio
    target_compile_options(arkade_knn PRIVATE /W4)
    set_target_properties(arkade_knn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
    )
else()
    # GCC/Clang
    target_compile_options(arkade_knn PRIVATE -Wall -Wextra)
    set_target_properties(arkade_knn PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

# ============================================================================
# INFORMACIÓN DE CONFIGURACIÓN
# ============================================================================

message(STATUS "===========================================")
message(STATUS "Configuración del proyecto Arkade GPU KNN")
message(STATUS "===========================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard: ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OptiX SDK: ${OptiX_INSTALL_DIR}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "FAISS Libraries: ${FAISS_LIBRARIES}")
message(STATUS "NMSLIB Libraries: ${NMSLIB_LIBRARIES}")
message(STATUS "FLANN Libraries: ${FLANN_LIBRARIES}")
message(STATUS "PTX Output: ${PTX_OUTPUT_DIR}")
message(STATUS "=========================================")
