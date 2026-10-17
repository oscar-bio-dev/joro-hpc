################################################################################
# JORO-HPC — CMake Build System (Arquitectura Modular por Capas)
# Simulador Mecanístico Multiescala LPDM-ABM para Trichonephila clavata
#
# Stack: C++20 · Intel oneAPI (icpx / DPC++ / SYCL) · OpenMP · AVX-512
#        NetCDF-C / HDF5 · PostgreSQL + PostGIS (libpqxx)
################################################################################
cmake_minimum_required(VERSION 3.22)

project(joro_hpc
    VERSION     0.1.0
    LANGUAGES   CXX
    DESCRIPTION "LPDM-ABM multi-scale simulator for Trichonephila clavata dispersal"
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Estándar C++20
# ─────────────────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD          20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Detección de compilador y banderas de aceleración
# ─────────────────────────────────────────────────────────────────────────────
option(JORO_ENABLE_AVX512 "Habilitar instrucciones AVX-512 (target Core i9-11900K / Rocket Lake)" OFF)
option(JORO_ENABLE_NATIVE "Habilitar optimizaciones de arquitectura del host (-march=native)" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    message(STATUS "[Compilador] Intel oneAPI DPC++ (icpx) detectado.")
    set(JORO_SYCL_FLAGS   "-fsycl")
    set(JORO_OPENMP_FLAGS "-qopenmp")
    if(JORO_ENABLE_AVX512)
        set(JORO_ARCH_FLAGS   "-march=rocketlake") # AVX-512 para Core i9-11900K
    elseif(JORO_ENABLE_NATIVE)
        set(JORO_ARCH_FLAGS   "-march=native")
    else()
        set(JORO_ARCH_FLAGS   "-march=x86-64-v3") # Portabilidad AVX2
    endif()
    add_compile_definitions(JORO_HAS_SYCL)
else()
    message(STATUS "[Compilador] Compilador estándar detectado: ${CMAKE_CXX_COMPILER_ID}.")
    set(JORO_SYCL_FLAGS   "")
    set(JORO_OPENMP_FLAGS "-fopenmp")
    if(JORO_ENABLE_AVX512)
        set(JORO_ARCH_FLAGS   "-mavx512f -mavx512bw")
    elseif(JORO_ENABLE_NATIVE)
        set(JORO_ARCH_FLAGS   "-march=native")
    else()
        set(JORO_ARCH_FLAGS   "-march=x86-64-v3") # Portabilidad segura para CI sin SIGILL
    endif()
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${JORO_ARCH_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -fsanitize=address")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dependencias externas
# ─────────────────────────────────────────────────────────────────────────────
find_package(OpenMP)
if(NOT OpenMP_FOUND)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "[OpenMP] Configurando OpenMP para Clang...")
        set(OpenMP_CXX_FLAGS "-fopenmp=libomp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "omp")
        find_package(OpenMP REQUIRED)
    else()
        find_package(OpenMP REQUIRED)
    endif()
endif()
find_package(PkgConfig REQUIRED)

# 3.a NetCDF-C para tensores ERA5 3D
pkg_check_modules(NETCDF IMPORTED_TARGET netcdf)
if(NETCDF_FOUND)
    message(STATUS "[NetCDF] Librería encontrada: ${NETCDF_VERSION}")
    add_compile_definitions(JORO_HAS_NETCDF)
else()
    message(WARNING "[NetCDF] 'netcdf' no encontrado vía pkg-config. Usando stubs de prueba.")
endif()

# 3.b libpqxx para conexión a PostgreSQL/PostGIS
pkg_check_modules(PQXX IMPORTED_TARGET libpqxx>=7.0)
if(PQXX_FOUND)
    message(STATUS "[PostgreSQL] libpqxx encontrada: ${PQXX_VERSION}")
    add_compile_definitions(JORO_HAS_LIBPQXX)
else()
    message(WARNING "[PostgreSQL] 'libpqxx' no encontrada vía pkg-config. Usando stubs de prueba.")
endif()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Biblioteca Interna del Núcleo: joro_core
# ─────────────────────────────────────────────────────────────────────────────
set(JORO_CORE_SOURCES
    src/io/era5_netcdf_reader.cpp
    src/io/postgis_client.cpp
    src/lpdm/lpdm_cpu_baseline.cpp
    src/lpdm/lpdm_sycl_kernel.cpp
    src/abm/abm_engine.cpp
    src/inference/morris_screening.cpp
    src/inference/simulated_annealing.cpp
)

add_library(joro_core STATIC ${JORO_CORE_SOURCES})

target_include_directories(joro_core PUBLIC
    "${CMAKE_SOURCE_DIR}/src"
)

target_compile_options(joro_core PUBLIC
    ${JORO_SYCL_FLAGS}
    ${JORO_OPENMP_FLAGS}
)

if(JORO_SYCL_FLAGS)
    target_link_options(joro_core PUBLIC ${JORO_SYCL_FLAGS})
endif()

target_link_libraries(joro_core PUBLIC
    OpenMP::OpenMP_CXX
)

if(NETCDF_FOUND)
    target_link_libraries(joro_core PUBLIC PkgConfig::NETCDF)
endif()

if(PQXX_FOUND)
    target_link_libraries(joro_core PUBLIC PkgConfig::PQXX)
endif()

# ─────────────────────────────────────────────────────────────────────────────
# 5. Ejecutable Principal: joro_hpc
# ─────────────────────────────────────────────────────────────────────────────
add_executable(joro_hpc src/main.cpp)
target_link_libraries(joro_hpc PRIVATE joro_core)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Suite de Pruebas Numéricas (tests/)
# ─────────────────────────────────────────────────────────────────────────────
enable_testing()

set(JORO_TESTS
    test_well_mixed
    test_euler_maruyama_conv
    test_cpu_gpu_equivalence
    test_parameter_recovery
)

foreach(test_name ${JORO_TESTS})
    add_executable(${test_name} tests/${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE joro_core)
    target_compile_options(${test_name} PRIVATE -UNDEBUG) # Garantizar aserciones científicas activas
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Benchmarks de Rendimiento HPC (benchmarks/)
# ─────────────────────────────────────────────────────────────────────────────
set(JORO_BENCHMARKS
    bench_usm_vs_buffers
    bench_abm_simd
)

foreach(bench_name ${JORO_BENCHMARKS})
    add_executable(${bench_name} benchmarks/${bench_name}.cpp)
    target_link_libraries(${bench_name} PRIVATE joro_core)
endforeach()

# ─────────────────────────────────────────────────────────────────────────────
# 8. Resumen de configuración
# ─────────────────────────────────────────────────────────────────────────────
message(STATUS "──────────────────────────────────────────────────────")
message(STATUS "  JORO-HPC v${PROJECT_VERSION} — Configuración CMake")
message(STATUS "  Compilador CXX : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Banderas SYCL  : ${JORO_SYCL_FLAGS}")
message(STATUS "  Banderas OpenMP: ${JORO_OPENMP_FLAGS}")
message(STATUS "  Banderas Arch  : ${JORO_ARCH_FLAGS}")
message(STATUS "  Librería Core  : joro_core (estática)")
message(STATUS "  Ejecutable     : joro_hpc")
message(STATUS "  Pruebas        : ${JORO_TESTS}")
message(STATUS "  Benchmarks     : ${JORO_BENCHMARKS}")
message(STATUS "──────────────────────────────────────────────────────")
