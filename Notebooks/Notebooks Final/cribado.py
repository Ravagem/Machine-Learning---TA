"""Cribado (D3): ¿aportan los grupos C (contexto del día) y P (planificación)?

Solo validación cruzada (4 pliegues del desarrollo); la prueba NO se usa.
XGBoost con los hiperparámetros que eligió el Entregable 2 para ese escenario, en k = 0 (F0) y k = 12 (F1).
Regla de aceptación (D1): mejora la media de validación Y gana en al menos 3 de 4 pliegues.
"""
import json, os, sys
import numpy as np
import pandas as pd
from prep import cargar, a_clase
from modelos import construir_features, ajustar, predecir, metricas_reg, metricas_clf

AQUI = os.path.dirname(os.path.abspath(__file__))
U_ADEL = float(sys.argv[1]) if len(sys.argv) > 1 else -1
U_RETR = float(sys.argv[2]) if len(sys.argv) > 2 else 5

d = cargar()
vm, Dm, FOLDS = d["vm"], d["Dm"], d["FOLDS"]
y_reg = vm["y"].values
y_clf = a_clase(y_reg, U_ADEL, U_RETR).astype(int)
mej = pd.read_csv(os.path.join(AQUI, "..", "Notebooks E2", "resultados", "resultados_cv_mejores.csv"))

GRUPOS = {"base": [], "+C": [d["C"]], "+P": [d["P"]], "+C+P": [d["C"], d["P"]], "+A": [d["A"]]}
filas = []
for tarea in ("reg", "clf"):
    for k, conj in ((0, "F0"), (12, "F1")):
        params = json.loads(mej[(mej.tarea == tarea) & (mej.modelo == "xgb") & (mej.k == k) & (mej.conjunto == conj)].params.iat[0])
        y = y_reg if tarea == "reg" else y_clf
        met = "MAE" if tarea == "reg" else "F1_macro"
        for g, extra in GRUPOS.items():
            X = construir_features(vm, Dm, k, conj, extra=extra)
            vals = []
            for a, b in FOLDS:
                m = ajustar(tarea, "xgb", params, X.iloc[a], y[a])
                p, pr = predecir(tarea, m, X.iloc[b])
                vals.append((metricas_reg(y[b], p) if tarea == "reg" else metricas_clf(y[b], p, pr))[met])
            filas.append({"tarea": tarea, "k": k, "conjunto": conj, "grupo": g, "metrica": met,
                          "media": np.mean(vals), "std": np.std(vals, ddof=1), **{f"p{j}": v for j, v in enumerate(vals, 1)}})
r = pd.DataFrame(filas)
# pliegues ganados frente a "base" (MAE: menor es mejor; F1: mayor es mejor)
for (t, k), g in r.groupby(["tarea", "k"]):
    base = g[g.grupo == "base"][["p1", "p2", "p3", "p4"]].values[0]
    for i in g.index:
        v = r.loc[i, ["p1", "p2", "p3", "p4"]].values.astype(float)
        mejor = (v < base) if t == "reg" else (v > base)
        r.loc[i, "pliegues_ganados"] = int(mejor.sum())
        r.loc[i, "dif_media"] = v.mean() - base.mean()
r["acepta_D1"] = np.where(r.tarea == "reg", r.dif_media < 0, r.dif_media > 0) & (r.pliegues_ganados >= 3) & (r.grupo != "base")
pd.set_option("display.width", 200)
print(f"Umbrales de clase: adelantado < {U_ADEL}, retrasado > {U_RETR}")
print(r.round(3).to_string(index=False))
r.to_csv(os.path.join(AQUI, "resultados", f"cribado_CP_{U_ADEL:g}_{U_RETR:g}.csv"), index=False)
