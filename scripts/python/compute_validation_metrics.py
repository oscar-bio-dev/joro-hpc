#!/usr/bin/env python3
"""
JORO-HPC · scripts/python/compute_validation_metrics.py
Calcula exhaustivamente todas las métricas oficiales de la validación fuera de muestra (2020–2024) para:
  - JORO-HPC M1 (MAP y Ensamble)
  - JORO-HPC M0 (K0 constante)
  - Kernel Exponencial Isotrópico (Null H2)
  - SDM Correlativo Estático (Null H1)

Genera el archivo maestro de métricas JSON y los datos para el informe final.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.linear_model import LinearRegression

PRED_CSV = "/mnt/hpc_scratch/joro-hpc/outputs/predictions_2020_2024_ensemble.csv"
OUT_JSON = "/mnt/hpc_scratch/joro-hpc/outputs/validation_metrics.json"

SEED_JSON = "/mnt/hpc_scratch/joro-hpc/outputs/seed_geometry.json"
PREV_JSON = "/mnt/hpc_scratch/joro-hpc/outputs/prevalence_scaling.json"

# Referencia heredada de la calibración: distancias del frente (P50/P90/P95) y envolvente de ocupación.
EPI_LAT = 34.13816
EPI_LON = -83.75723


def load_seed_geometry():
    """Centroide y radio del polígono semilla (artefacto de `joro_evaluate`; referencia angular del frente)."""
    if not os.path.exists(SEED_JSON):
        raise SystemExit(f"Falta {SEED_JSON}: ejecute joro_evaluate primero")
    with open(SEED_JSON) as fh:
        return json.load(fh)


SEED = load_seed_geometry()


def dist_km(lat, lon):
    dlat = (lat - EPI_LAT) * 111.0
    dlon = (lon - EPI_LON) * 111.0 * np.cos(np.radians(EPI_LAT))
    return np.sqrt(dlat**2 + dlon**2)


def dist_seed_km(lat, lon):
    dlat = (lat - SEED["centroid_lat"]) * 111.0
    dlon = (lon - SEED["centroid_lon"]) * 111.0 * np.cos(np.radians(SEED["centroid_lat"]))
    return np.sqrt(dlat**2 + dlon**2)


def azimuth_deg(lat, lon):
    """Azimut (0 = Norte, 90 = Este) respecto al centroide geométrico del polígono semilla."""
    dlat = (lat - SEED["centroid_lat"]) * 111.0
    dlon = (lon - SEED["centroid_lon"]) * 111.0 * np.cos(np.radians(SEED["centroid_lat"]))
    deg = np.degrees(np.arctan2(dlon, dlat)) % 360.0
    return deg

def circular_mean(angles_deg, weights=None):
    rads = np.radians(angles_deg)
    if weights is None:
        s = np.sin(rads).mean()
        c = np.cos(rads).mean()
    else:
        w_sum = np.sum(weights)
        if w_sum <= 0: return 0.0
        s = np.sum(weights * np.sin(rads)) / w_sum
        c = np.sum(weights * np.cos(rads)) / w_sum
    res = np.degrees(np.arctan2(s, c)) % 360.0
    return res

def angular_diff(a1, a2):
    diff = abs(a1 - a2) % 360.0
    return diff if diff <= 180.0 else 360.0 - diff

def get_calibration_slope_intercept(p, y):
    eps = 1e-6
    p_c = np.clip(p, eps, 1.0 - eps)
    logit_p = np.log(p_c / (1.0 - p_c)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(logit_p, y)
    return float(lr.coef_[0]), float(lr.intercept_)

def calc_balanced_brier(p, y, dists, radius=350.0):
    pos_mask = (y == 1)
    neg_mask = (y == 0) & (dists <= radius)
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return float(brier_score_loss(y, p))
    b_pos = np.mean((1.0 - p[pos_mask]) ** 2)
    b_neg = np.mean((p[neg_mask]) ** 2)
    return float(0.5 * (b_pos + b_neg))

def main():
    print("▶ Cargando predicciones de ensamble (validación 2020–2024)...")
    df = pd.read_csv(PRED_CSV)
    print(f"✔ Cargadas {len(df)} filas celda-año (2020–2024).")

    df["dist"] = dist_km(df["centroid_lat"].values, df["centroid_lon"].values)
    df["angle"] = azimuth_deg(df["centroid_lat"].values, df["centroid_lon"].values)
    df["dist_seed"] = dist_seed_km(df["centroid_lat"].values, df["centroid_lon"].values)
    seed_r = float(SEED["radius_km"])

    models = {
        "m1_map": "p_m1_map",
        "m1_median": "p_m1_median",
        "m0_map": "p_m0_map",
        "iso_null": "p_iso_null",
        "sdm": "p_sdm"
    }

    results = {
        "global": {},
        "by_year": {},
        "kinematics": {},
        "thermal_persistence": {},
        "ecoregions": {},
        "uncertainty": {}
    }

    y_all = df["y_observed"].values
    dists_all = df["dist"].values

    print("\n--- 1. DESEMPEÑO PROBABILÍSTICO GLOBAL (2020-2024) ---")
    for m_key, col in models.items():
        p_vals = df[col].values
        bg = float(brier_score_loss(y_all, p_vals))
        bb = calc_balanced_brier(p_vals, y_all, dists_all, 350.0)
        ll = float(log_loss(y_all, np.clip(p_vals, 1e-6, 1.0 - 1e-6)))
        auc = float(roc_auc_score(y_all, p_vals))
        prauc = float(average_precision_score(y_all, p_vals))
        slope, intercept = get_calibration_slope_intercept(p_vals, y_all)

        results["global"][m_key] = {
            "brier_global": bg,
            "brier_balanced": bb,
            "log_loss": ll,
            "auc_roc": auc,
            "pr_auc": prauc,
            "calib_slope": slope,
            "calib_intercept": intercept
        }
        print(f"[{m_key:10s}] Brier: {bg:.5f} (Bal: {bb:.5f}) | LogLoss: {ll:.4f} | AUC: {auc:.4f} | PR-AUC: {prauc:.4f} | Slope: {slope:.3f}")

    print("\n--- 2. DESGLOSE ANUAL FUERA DE MUESTRA (2020-2024) ---")
    for yr in range(2020, 2025):
        sub = df[df["year"] == yr]
        y_yr = sub["y_observed"].values
        dists_yr = sub["dist"].values
        yr_res = {"presences": int(y_yr.sum())}

        for m_key, col in models.items():
            p_vals = sub[col].values
            bg = float(brier_score_loss(y_yr, p_vals))
            bb = calc_balanced_brier(p_vals, y_yr, dists_yr, 350.0)
            ll = float(log_loss(y_yr, np.clip(p_vals, 1e-6, 1.0 - 1e-6)))
            auc = float(roc_auc_score(y_yr, p_vals))
            prauc = float(average_precision_score(y_yr, p_vals))
            yr_res[m_key] = {"brier": bg, "brier_bal": bb, "log_loss": ll, "auc": auc, "pr_auc": prauc}

        results["by_year"][yr] = yr_res
        print(f"Año {yr} (N_pos={yr_res['presences']:3d}) | Brier M1: {yr_res['m1_map']['brier']:.5f}, Kern: {yr_res['iso_null']['brier']:.5f}, SDM: {yr_res['sdm']['brier']:.5f} | AUC M1: {yr_res['m1_map']['auc']:.4f}, Kern: {yr_res['iso_null']['auc']:.4f}, SDM: {yr_res['sdm']['auc']:.4f}")

    print("\n--- 3. CINEMÁTICA Y GEOMETRÍA DEL FRENTE DE INVASIÓN (H2) ---")
    kin = {}
    for yr in range(2020, 2025):
        sub = df[df["year"] == yr]
        obs = sub[sub["y_observed"] == 1]
        obs_dists = obs["dist"].values
        obs_angles = obs[obs["dist_seed"] > seed_r]["angle"].values   # solo presencias fuera del radio semilla

        o_p50 = float(np.percentile(obs_dists, 50))
        o_p90 = float(np.percentile(obs_dists, 90))
        o_p95 = float(np.percentile(obs_dists, 95))
        o_ang = float(circular_mean(obs_angles)) if len(obs_angles) else float("nan")

        yr_kin = {
            "obs_p50": o_p50, "obs_p90": o_p90, "obs_p95": o_p95, "obs_angle": o_ang
        }

        for m_key, col in [("m1_map", "p_m1_map"), ("m0_map", "p_m0_map"), ("iso_null", "p_iso_null")]:
            # Celdas con presencia biológica activa predicha (colonización activa: p > 0.001)
            pred_cells = sub[sub[col] > 0.001]
            if len(pred_cells) == 0:
                pred_cells = sub[sub[col] > 0.0]
            p_dists = pred_cells["dist"].values
            p_angles = pred_cells[pred_cells["dist_seed"] > seed_r]["angle"].values

            m_p50 = float(np.percentile(p_dists, 50)) if len(p_dists) else 0.0
            m_p90 = float(np.percentile(p_dists, 90)) if len(p_dists) else 0.0
            m_p95 = float(np.percentile(p_dists, 95)) if len(p_dists) else 0.0
            m_ang = float(circular_mean(p_angles)) if len(p_angles) else float("nan")
            rmse_front = float(np.sqrt(np.mean([(m_p50 - o_p50)**2, (m_p90 - o_p90)**2, (m_p95 - o_p95)**2])))
            ang_err = float(angular_diff(m_ang, o_ang))

            yr_kin[m_key] = {
                "p50": m_p50, "p90": m_p90, "p95": m_p95,
                "angle": m_ang, "rmse_front_km": rmse_front, "angular_error_deg": ang_err
            }

        kin[yr] = yr_kin
        print(f"Año {yr} | Obs P95: {o_p95:.1f} km, M1 P95: {yr_kin['m1_map']['p95']:.1f} km, Iso P95: {yr_kin['iso_null']['p95']:.1f} km | Error Ang M1: {yr_kin['m1_map']['angular_error_deg']:.1f}°, Iso: {yr_kin['iso_null']['angular_error_deg']:.1f}°")

    sim_kin_path = "/mnt/hpc_scratch/joro-hpc/outputs/validation_sim_kinematics.json"
    if os.path.exists(sim_kin_path):
        with open(sim_kin_path) as kf:
            sim_kin = json.load(kf)
        for yr_str, yk in sim_kin.items():
            yr_int = int(yr_str)
            if yr_int in kin:
                # Dirección alternativa: celdas con ocupación latente predicha (psi*p_det > 0.001), antes de sustituirla
                # por la cinemática del frente biológico (celdas con adultas > 2) que exporta C++.
                for _mk in ("m1_map", "iso_null"):
                    kin[yr_int][_mk]["angle_psi"] = kin[yr_int][_mk]["angle"]
                    kin[yr_int][_mk]["angular_error_psi_deg"] = kin[yr_int][_mk]["angular_error_deg"]
                kin[yr_int]["obs_angle_psi"] = kin[yr_int]["obs_angle"]
                kin[yr_int]["obs_p95"] = yk["obs_p95"]
                kin[yr_int]["obs_angle"] = yk["obs_angle"]
                kin[yr_int]["m1_map"]["p95"] = yk["m1_map"]["p95"]
                kin[yr_int]["m1_map"]["angle"] = yk["m1_map"]["angle"]
                kin[yr_int]["m1_map"]["angular_error_deg"] = yk["m1_map"]["angular_error_deg"]
                kin[yr_int]["iso_null"]["p95"] = yk["iso_null"]["p95"]
                kin[yr_int]["iso_null"]["angle"] = yk["iso_null"]["angle"]
                kin[yr_int]["iso_null"]["angular_error_deg"] = yk["iso_null"]["angular_error_deg"]
        print("✔ Cinemática de simulación C++ acoplada e integrada exitosamente.")

    results["kinematics"] = kin

    # Error en fecha de primera llegada por celda
    cells_with_presence = df[df["y_observed"] == 1]["cell_id"].unique()
    first_obs = df[df["y_observed"] == 1].groupby("cell_id")["year"].min()

    first_errs = {"m1_map": [], "iso_null": [], "sdm": []}
    for cid in cells_with_presence:
        t_obs = first_obs[cid]
        c_rows = df[df["cell_id"] == cid].sort_values("year")
        for m_key, col in [("m1_map", "p_m1_map"), ("iso_null", "p_iso_null"), ("sdm", "p_sdm")]:
            thresh = 0.05
            pred_arrival = c_rows[c_rows[col] >= thresh]
            t_pred = pred_arrival["year"].min() if len(pred_arrival) > 0 else 2025
            first_errs[m_key].append(abs(t_pred - t_obs))

    results["first_arrival_mae"] = {
        k: float(np.mean(v)) for k, v in first_errs.items()
    }
    print(f"Error MAE Primera Llegada: M1={results['first_arrival_mae']['m1_map']:.2f} años, Iso={results['first_arrival_mae']['iso_null']:.2f} años, SDM={results['first_arrival_mae']['sdm']:.2f} años")

    print("\n--- 4. PERSISTENCIA TÉRMICA Y TRANSFERENCIA POR ECORREGIONES (H1) ---")
    # Umbral de corte binario óptimo tau* determinado exclusivamente en 2014–2019
    # mediante maximización del índice J de Youden / TSS
    tau_opt = {
        "m1_map": 0.001208,
        "m1_median": 0.001208,
        "m0_map": 0.001208,
        "iso_null": 0.001208,
        "sdm": 0.001208
    }
    # Zona norte de restricción invernal (lat >= 38.0°N / límite ecorregional norte)
    north_mask = (df["centroid_lat"] >= 38.0)
    fpr_north = {}
    for m_key, col in models.items():
        sub_north = df[north_mask]
        neg_north = sub_north[sub_north["y_observed"] == 0]
        thresh = tau_opt.get(m_key, 0.001208)
        fp = (neg_north[col] >= thresh).sum()
        fpr = float(fp / len(neg_north)) if len(neg_north) > 0 else 0.0
        fpr_north[m_key] = fpr

    results["thermal_persistence"]["fpr_north_40"] = fpr_north
    results["thermal_persistence"]["tau_opt"] = tau_opt
    print(f"Tasa Falsos Positivos Norte (>38°–40°N, tau*={0.001208}): M1={fpr_north['m1_map']*100:.2f}%, M0={fpr_north['m0_map']*100:.2f}%, Iso={fpr_north['iso_null']*100:.2f}%, SDM={fpr_north['sdm']*100:.2f}%")

    # Persistencia interanual (>= 2 años consecutivos)
    # Comparar la probabilidad de persistencia condicional a haber estado ocupado el año anterior
    # en celdas con registros en t-1
    cells_seen = set()
    persist_records = {"observed": [], "m1": [], "m0": [], "iso": [], "sdm": []}
    for yr in range(2020, 2025):
        sub = df[df["year"] == yr]
        for _, row in sub.iterrows():
            cid = row["cell_id"]
            if cid in cells_seen:
                persist_records["observed"].append(row["y_observed"])
                persist_records["m1"].append(row["p_m1_map"])
                persist_records["m0"].append(row["p_m0_map"])
                persist_records["iso"].append(row["p_iso_null"])
                persist_records["sdm"].append(row["p_sdm"])
            if row["y_observed"] == 1:
                cells_seen.add(cid)

    if persist_records["observed"]:
        y_p = np.array(persist_records["observed"])
        results["thermal_persistence"]["interannual_persistence_brier"] = {
            "m1": float(brier_score_loss(y_p, persist_records["m1"])),
            "m0": float(brier_score_loss(y_p, persist_records["m0"])),
            "iso": float(brier_score_loss(y_p, persist_records["iso"])),
            "sdm": float(brier_score_loss(y_p, persist_records["sdm"])),
            "n_evaluations": len(y_p)
        }
        print(f"Brier Persistencia Interanual (N={len(y_p)}): M1={results['thermal_persistence']['interannual_persistence_brier']['m1']:.4f}, M0={results['thermal_persistence']['interannual_persistence_brier']['m0']:.4f}, Iso={results['thermal_persistence']['interannual_persistence_brier']['iso']:.4f}, SDM={results['thermal_persistence']['interannual_persistence_brier']['sdm']:.4f}")

    # Desempeño por bloques ecorregionales (Bloque 5 visto vs Bloques no vistos)
    eco_res = {}
    for block_id in sorted(df["ecoregion_block"].unique()):
        sub_b = df[df["ecoregion_block"] == block_id]
        y_b = sub_b["y_observed"].values
        pres_b = int(y_b.sum())
        b_dict = {"n_rows": len(sub_b), "presences": pres_b}
        if pres_b > 0 and len(np.unique(y_b)) > 1:
            for m_key, col in models.items():
                p_b = sub_b[col].values
                b_dict[m_key] = {
                    "brier": float(brier_score_loss(y_b, p_b)),
                    "auc": float(roc_auc_score(y_b, p_b))
                }
        else:
            for m_key, col in models.items():
                p_b = sub_b[col].values
                b_dict[m_key] = {
                    "brier": float(brier_score_loss(y_b, p_b)),
                    "auc": 0.5
                }
        eco_res[int(block_id)] = b_dict

    results["ecoregions"] = eco_res
    b5 = eco_res.get(5, {})
    print(f"Bloque 5 (Visto en Calibración): N_pos={b5.get('presences',0)} | Brier M1={b5.get('m1_map',{}).get('brier',0):.5f}, SDM={b5.get('sdm',{}).get('brier',0):.5f} | AUC M1={b5.get('m1_map',{}).get('auc',0):.4f}, SDM={b5.get('sdm',{}).get('auc',0):.4f}")

    # Bloque 5 (Visto)
    sub_5 = df[df["ecoregion_block"] == 5]
    y_5 = sub_5["y_observed"].values
    brier_seen = {}
    auc_seen = {}
    for m_key, col in models.items():
        p_5 = sub_5[col].values
        brier_seen[m_key] = float(brier_score_loss(y_5, p_5))
        auc_seen[m_key] = float(roc_auc_score(y_5, p_5)) if len(np.unique(y_5)) > 1 else 0.5

    # Bloques no vistos
    unseen_df = df[df["ecoregion_block"] != 5]
    y_unseen = unseen_df["y_observed"].values
    brier_unseen = {}
    auc_unseen = {}
    print(f"Bloques Nuevos (No vistos, N_pos={y_unseen.sum()}):")
    for m_key, col in models.items():
        p_u = unseen_df[col].values
        b_u = float(brier_score_loss(y_unseen, p_u))
        auc_u = float(roc_auc_score(y_unseen, p_u)) if y_unseen.sum() > 0 else 0.5
        brier_unseen[m_key] = b_u
        auc_unseen[m_key] = auc_u
        print(f"  [{m_key:10s}] Brier: {b_u:.5f} | AUC: {auc_u:.4f}")

    results["thermal_persistence"]["brier_seen"] = brier_seen
    results["thermal_persistence"]["auc_seen"] = auc_seen
    results["thermal_persistence"]["brier_unseen"] = brier_unseen
    results["thermal_persistence"]["auc_unseen"] = auc_unseen

    print("\n--- 5. PROPAGACIÓN DE INCERTIDUMBRE (Theta_comp) ---")

    # Cobertura agregada por año: percentiles de los TOTALES ANUALES de cada miembro del ensamble Θ_comp.
    # `joro_evaluate` exporta, por miembro, la suma anual de probabilidades (cruda) y la ajustada con el offset
    # logit de prevalencia ajustado exclusivamente en 2014–2019 (outputs/prevalence_scaling.json).
    if not os.path.exists(PREV_JSON):
        raise SystemExit(f"Falta {PREV_JSON}: ejecute joro_evaluate primero")
    with open(PREV_JSON) as fh:
        prev = json.load(fh)
    raw_t = np.array(prev["ensemble"]["raw_totals"])      # [miembros, 5]
    adj_t = np.array(prev["ensemble"]["adj_totals"])
    yearly_cov = {}
    for k, yr in enumerate(range(2020, 2025)):
        obs_sum = int(df[df["year"] == yr]["y_observed"].sum())
        assert obs_sum == int(round(prev["evaluation"]["observed"][k])), "observados inconsistentes con prevalence_scaling.json"
        r5, rm, r95 = (float(np.percentile(raw_t[:, k], q)) for q in (5, 50, 95))
        a5, am, a95 = (float(np.percentile(adj_t[:, k], q)) for q in (5, 50, 95))
        yearly_cov[yr] = {
            "obs": obs_sum,
            "p5": r5, "median": rm, "p95": r95, "in_90_pi": bool(r5 <= obs_sum <= r95),
            "adj_p5": a5, "adj_median": am, "adj_p95": a95, "adj_in_90_pi": bool(a5 <= obs_sum <= a95),
            "m1_map_raw": float(prev["evaluation"]["m1_map"]["raw"][k]),
            "m1_map_adj": float(prev["evaluation"]["m1_map"]["adj"][k]),
        }
        print(f"Año {yr}: Obs={obs_sum:4d} | crudo P5-P95: [{r5:6.1f}, {r95:6.1f}] ({yearly_cov[yr]['in_90_pi']}) | "
              f"ajustado P5-P95: [{a5:6.1f}, {a95:6.1f}] ({yearly_cov[yr]['adj_in_90_pi']})")
    p5 = df["p_m1_p5"].values
    p95 = df["p_m1_p95"].values
    med = df["p_m1_median"].values

    try:
        with open("/mnt/hpc_scratch/joro-hpc/checkpoints/calibration_summary_M1.json") as fh:
            ensemble_size = int(json.load(fh)["compatible_members"])
    except (OSError, KeyError, ValueError):
        ensemble_size = None
    results["uncertainty"] = {
        "ensemble_size": ensemble_size,
        "mean_band_width": float(np.mean(p95 - p5)),
        "yearly_aggregated_coverage": yearly_cov,
        "prevalence": {
            "method": prev["method"],
            "calibration_observed": prev["calibration"]["observed"],
            "m1_map": prev["calibration"]["m1_map"],
            "m0_map": prev["calibration"]["m0_map"],
            "ensemble_delta_median": float(np.median(prev["ensemble"]["delta"])),
            "ensemble_delta_p5": float(np.percentile(prev["ensemble"]["delta"], 5)),
            "ensemble_delta_p95": float(np.percentile(prev["ensemble"]["delta"], 95)),
            "ensemble_calib_raw_median": float(np.median(prev["ensemble"]["calib_raw_expected"])),
            "ensemble_calib_adj_median": float(np.median(prev["ensemble"]["calib_adj_expected"])),
        },
        "seed_geometry": SEED,
    }

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✔ Métricas maestras guardadas en: {OUT_JSON}")

if __name__ == "__main__":
    main()
