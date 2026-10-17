#pragma once

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <string>

/// Macro de aserción científica que no se desactiva bajo -DNDEBUG
/// Garantiza que los tests en modo Release verifiquen realmente la física y estadística.
#define JORO_TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[TEST FAILED] " << __FILE__ << ":" << __LINE__ \
                      << " en función '" << __func__ << "'\n" \
                      << "  Condición fallida : " << #cond << "\n" \
                      << "  Diagnóstico       : " << msg << "\n\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

#define JORO_TEST_ASSERT_NEAR(val, target, eps, msg) \
    do { \
        double _diff = std::abs(static_cast<double>(val) - static_cast<double>(target)); \
        if (_diff > static_cast<double>(eps)) { \
            std::cerr << "\n[TEST FAILED] " << __FILE__ << ":" << __LINE__ \
                      << " en función '" << __func__ << "'\n" \
                      << "  Valor obtenido : " << (val) << "\n" \
                      << "  Valor esperado : " << (target) << " (diff=" << _diff << " > tol=" << (eps) << ")\n" \
                      << "  Diagnóstico    : " << msg << "\n\n"; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)
