"""Búsqueda en malla de la entrega final (validación cruzada temporal, 4 pliegues del desarrollo).

Igual que la etapa 8 del Entregable 2, con dos cambios fijados en PROTOCOLO.md:
  D5: malla AMPLIADA de XGBoost (agrega profundidad 2, tasa 0,01 y 1 600 árboles; contiene la malla original).
  D4: la clasificación se corre con los umbrales −1/+5 (principal) y −3/+10 (análisis de sensibilidad).
Reproducibilidad: cada modelo usa 1 hilo (resultado determinista en una misma máquina) y se paraleliza entre
combinaciones. Guarda cada combinación en resultados/cv_raw.csv apenas termina y retoma si se interrumpe.

Uso: python experimento.py [n_procesos]
"""
import json, os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid
from prep import cargar, a_clase
from modelos import construir_features, ajustar, predecir_variantes, metricas_reg, metricas_clf, MODO_X

AQUI = os.path.dirname(os.path.abspath(__file__))
ARCH = os.path.join(AQUI, "resultados", "cv_raw.csv")
ESCENARIOS = [(0, "F0"), (4, "F1"), (8, "F1"), (12, "F1"), (4, "F2"), (8, "F2"), (12, "F2")]
UMBRALES = {"cao": (-1, 5), "amplio": (-3, 10)}

XGB_ORIG = {"n_estimators": [200, 400, 800], "learning_rate": [0.03, 0.1], "max_depth": [3, 4, 6], "subsample": [0.8, 1.0]}
XGB_AMPL = {"n_estimators": [200, 400, 800, 1600], "learning_rate": [0.01, 0.03, 0.1], "max_depth": [2, 3, 4, 6],
            "subsample": [0.8, 1.0]}
GRIDS = {
    "reg": {"ridge": {"alpha": [0.01, 0.1, 1, 10, 100]},
            "rf": {"n_estimators": [200, 500], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 5, 10]},
            "xgb": XGB_AMPL},
    "clf": {"logistic": {"C": [0.01, 0.1, 1, 10, 100], "class_weight": [None, "balanced"]},
            "rf": {"n_estimators": [200, 500], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 5, 10],
                   "class_weight": [None, "balanced"]},
            "xgb": {**XGB_AMPL, "class_weight": [None, "balanced"]}},
}


COLS = (["clave", "tarea", "umbral", "modelo", "k", "conjunto", "params", "malla_original", "metrica", "metrica_media",
         "metrica_std", "pliegue1", "pliegue2", "pliegue3", "pliegue4", "segundos"]
        + [f"{c}_media" for c in ["RMSE", "R2", "recall_retrasado", "exactitud", "exactitud_bal", "sens_adel", "espec_adel",
                                  "sens_punt", "espec_punt", "sens_retr", "espec_retr", "PR_AUC"]])


def en_malla_original(modelo, p):
    """True si la combinación pertenece a la malla del Entregable 2 (para reportar original vs. ampliada)."""
    if modelo != "xgb":
        return True
    return all(p[c] in XGB_ORIG[c] for c in XGB_ORIG)


def clave(tarea, umbral, modelo, k, conj, p):
    return f"{tarea}|{umbral}|{modelo}|{k}|{conj}|{json.dumps(p, sort_keys=True)}"


def trabajos():
    out = []
    for tarea, umbrales in (("reg", ["-"]), ("clf", list(UMBRALES))):
        for umbral in umbrales:
            for k, conj in ESCENARIOS:
                for modelo, grid in GRIDS[tarea].items():
                    n_list = grid.get("n_estimators", [None])
                    resto = {a: b for a, b in grid.items() if a != "n_estimators"}
                    for params in ParameterGrid(resto):
                        out.append((tarea, umbral, modelo, k, conj, params, n_list))
    return out


_D = None


def _datos():
    global _D
    if _D is None:
        _D = cargar()
    return _D


def correr(tarea, umbral, modelo, k, conj, params, n_list):
    d = _datos()
    vm, Dm = d["vm"], d["Dm"]
    gps = d["GPS_K"][k] if conj == "F2" else None
    X = construir_features(vm, Dm, k, conj, gps, **MODO_X[modelo])
    y = vm["y"].values if tarea == "reg" else a_clase(vm["y"].values, *UMBRALES[umbral]).astype(int)
    t0 = time.time()
    fit_p = dict(params) if n_list == [None] else {**params, "n_estimators": max(n_list)}
    por_n = {n: [] for n in n_list}
    for a, b in d["FOLDS"]:
        m = ajustar(tarea, modelo, fit_p, X.iloc[a], y[a], n_jobs=1)
        for n, (pred, proba) in predecir_variantes(tarea, modelo, m, X.iloc[b], n_list).items():
            por_n[n].append(metricas_reg(y[b], pred) if tarea == "reg" else metricas_clf(y[b], pred, proba))
    filas = []
    met = "MAE" if tarea == "reg" else "F1_macro"
    for n, mets in por_n.items():
        p = dict(params) if n is None else {**params, "n_estimators": n}
        vals = [mm[met] for mm in mets]
        fila = {"clave": clave(tarea, umbral, modelo, k, conj, p), "tarea": tarea, "umbral": umbral, "modelo": modelo,
                "k": k, "conjunto": conj, "params": json.dumps(p, sort_keys=True), "malla_original": en_malla_original(modelo, p),
                "metrica": met, "metrica_media": np.mean(vals), "metrica_std": np.std(vals, ddof=1),
                **{f"pliegue{j}": v for j, v in enumerate(vals, 1)}, "segundos": (time.time() - t0) / len(n_list)}
        for c in mets[0]:
            if c != met:
                fila[f"{c}_media"] = np.mean([mm[c] for mm in mets])
        filas.append(fila)
    return filas


if __name__ == "__main__":
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    _datos()
    todos = trabajos()
    hechos = set(pd.read_csv(ARCH).clave) if os.path.exists(ARCH) else set()
    pend = [w for w in todos if any(clave(*w[:5], {**w[5], **({} if n is None else {"n_estimators": n})}) not in hechos for n in w[6])]
    # los más lentos primero (XGBoost clasificación) para equilibrar la carga
    pend.sort(key=lambda w: (w[2] != "xgb", w[0] != "clf"))
    n_comb = sum(len(w[6]) for w in todos)
    print(f"Combinaciones: {n_comb} | entrenamientos: {len(todos)} x 4 pliegues | pendientes: {len(pend)}", flush=True)
    t_ini, hechos_n = time.time(), 0
    lote = 64
    for i in range(0, len(pend), lote):
        res = Parallel(n_jobs=n_proc)(delayed(correr)(*w) for w in pend[i:i + lote])
        filas = [f for r in res for f in r]
        df = pd.DataFrame(filas).reindex(columns=COLS)
        df.to_csv(ARCH, mode="a", header=not os.path.exists(ARCH), index=False)
        hechos_n += len(res)
        tr = time.time() - t_ini
        print(f"[{hechos_n}/{len(pend)}] {tr/60:.1f} min | faltan ~{tr/hechos_n*(len(pend)-hechos_n)/60:.0f} min", flush=True)
    print("Búsqueda terminada", flush=True)
