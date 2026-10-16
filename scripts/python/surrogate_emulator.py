#!/usr/bin/env python3
"""
JORO-HPC  ·  scripts/python/surrogate_emulator.py
Plan B: Emulador subrogado (Gaussian Process / Random Forest) sobre Latin Hypercube.
Sección 13.3 de JORO-HPC.tex.

Se activa si el tiempo proyectado de calibración supera los 15 días continuos
o el costo por evaluación excede los 300 segundos.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger("joro.emulator")


def generate_latin_hypercube_design(
    bounds: Dict[str, Tuple[float, float]],
    n_samples: int = 500,
    seed: int = 42819,
) -> np.ndarray:
    """Genera una muestra de diseño por Hipercubo Latino (LHS) en el espacio de parámetros."""
    n_dim = len(bounds)
    sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
    sample_unit = sampler.random(n=n_samples)

    lower_bounds = np.array([b[0] for b in bounds.values()])
    upper_bounds = np.array([b[1] for b in bounds.values()])
    scaled_samples = qmc.scale(sample_unit, lower_bounds, upper_bounds)

    logger.info("Generado diseño LHS de %d puntos en %d dimensiones.", n_samples, n_dim)
    return scaled_samples


def train_surrogate_models(
    X_samples: np.ndarray,
    Y_costs: np.ndarray,
    model_type: str = "gp",
) -> Tuple[object, Dict[str, float]]:
    """Entrena un emulador para aproximar la función de costo J(theta).

    Parameters
    ----------
    X_samples : np.ndarray
        Matriz de parámetros (N, d).
    Y_costs : np.ndarray
        Vector de costo multicriterio J(theta) (N,).
    model_type : str
        'gp' para Gaussian Process con kernel Matern 5/2, o 'rf' para Random Forest.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_samples, Y_costs, test_size=0.2, random_state=42
    )

    if model_type == "gp":
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-3)
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)
    else:
        model = RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)),
    }
    logger.info("Entrenamiento del emulador (%s) finalizado: RMSE=%.4f, R²=%.4f", model_type, metrics["rmse"], metrics["r2"])
    return model, metrics
