"""Genera Demo/index.html: demostración interactiva (un solo archivo, funciona sin internet).

Todos los valores se calculan aquí con los modelos XGBoost entrenados con el desarrollo (los mismos evaluados en el
paper) y se incrustan como JSON en la plantilla Demo/plantilla.html. El HTML no ejecuta ningún modelo: muestra
resultados precalculados.

Uso (desde la raíz del repositorio):  python Demo/construir_demo.py
"""
import json, os, sys
import numpy as np
import pandas as pd
import joblib

AQUI = os.path.dirname(os.path.abspath(__file__))
FINAL = os.path.join(AQUI, "..", "Notebooks", "Notebooks Final")
sys.path.insert(0, FINAL)
from prep import cargar, DATA_DIR, a_clase  # noqa: E402
from modelos import construir_features      # noqa: E402

d = cargar()
vm, Dm, S, E_ok, ok = d["vm"], d["Dm"], d["S"][d["ok"]], d["E_ok"][d["ok"]], d["ok"]
IDX_TEST = d["IDX_TEST"]
MOD = lambda f: joblib.load(os.path.join(FINAL, "modelos", f))
r2 = lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 2)

# ---------------------------------------------------------------- puntos de control y traza de la ruta
PC = [(-13.5552, -71.8568, "Inicio de la vuelta (extremo este)"), (-13.5535, -71.8606, "Punto concurrido"),
      (-13.5445, -71.8883, "Punto concurrido"), (-13.5288, -71.9372, "Av. de la Cultura"),
      (-13.5216, -71.9651, "Av. de la Cultura, cerca del centro"), (-13.5256, -71.9833, "Punto concurrido"),
      (-13.5259, -71.9827, "Punto concurrido"), (-13.5359, -71.9804, "Punto concurrido"),
      (-13.5441, -71.9821, "Punto concurrido"), (-13.5543, -71.9908, "Punto más occidental (retorno)"),
      (-13.5419, -71.9821, "Punto concurrido"), (-13.5355, -71.9804, "Punto concurrido"),
      (-13.5226, -71.9818, "Mercado San Pedro"), (-13.5233, -71.9728, "Av. Tullumayo"),
      (-13.5226, -71.9714, "Punto concurrido"), (-13.5423, -71.8991, "Av. de la Cultura"),
      (-13.5441, -71.8888, "Punto concurrido"), (-13.5529, -71.8648, "Fin de la vuelta (extremo este)")]

# traza: registros GPS de una vuelta de prueba completa (todos los PC con hora real válida)
completa = [i for i in IDX_TEST if not np.isnan(E_ok[i]).any()][0]
veh, fecha = vm.vehicle_id_id.iat[completa], vm.date.iat[completa]
mon = pd.read_csv(os.path.join(DATA_DIR, "Monitoring.csv"), dtype={"vehicle_id_id": str},
                  usecols=["vehicle_id_id", "date", "time", "latitude", "longitude"])
mon = mon[(mon.vehicle_id_id == veh) & (mon.date == fecha)].copy()
mon["t"] = pd.to_timedelta(mon.time).dt.total_seconds()
mon = mon[(mon.t >= E_ok[completa, 0]) & (mon.t <= E_ok[completa, 17])].sort_values("t").drop_duplicates("t")
traza = mon[["latitude", "longitude"]].round(5).values.tolist()
print(f"Traza: vuelta {vm.id.iat[completa]} ({len(traza)} puntos GPS)")

# ---------------------------------------------------------------- desviación media por PC y franja (4 004 vueltas)
FRANJAS = [("todas", "Todas las horas", 0, 30), ("madrugada", "Madrugada (4–6 h)", 4, 6), ("manana", "Mañana (6–10 h)", 6, 10),
           ("mediodia", "Mediodía (10–14 h)", 10, 14), ("tarde", "Tarde y noche (14+ h)", 14, 30)]
perfil = {}
for clave, nom, a, b in FRANJAS:
    m = ((vm.hora_salida >= a) & (vm.hora_salida < b)).values
    perfil[clave] = {"nombre": nom, "n": int(m.sum()), "media": [r2(v) for v in np.nanmean(Dm[m], axis=0)]}
for dia, nom in ((5, "Solo sábados"),):
    m = (vm.dia_semana == dia).values
    perfil["sabado"] = {"nombre": nom, "n": int(m.sum()), "media": [r2(v) for v in np.nanmean(Dm[m], axis=0)]}

# ---------------------------------------------------------------- vueltas de prueba: real, predicción por k, clase y SHAP
X = {}
for k, c in ((0, "F0"), (4, "F1"), (8, "F1"), (12, "F1")):
    X[k] = construir_features(vm, Dm, k, c)
pred_reg, pred_clf, proba_clf = {}, {}, {}
for k, c in ((0, "F0"), (4, "F1"), (8, "F1"), (12, "F1")):
    a = MOD(f"xgb_reg_k{k}_{c}.joblib")
    pred_reg[k] = a["modelo"].predict(X[k][a["columnas"]].iloc[IDX_TEST])
    a = MOD(f"xgb_clf_k{k}_{c}.joblib")
    pr = a["modelo"].predict_proba(X[k][a["columnas"]].iloc[IDX_TEST])
    w0, w2, acepta, _ = a["pesos_clase"]
    prw = pr * np.array([w0, 1, w2]) if acepta else pr       # pesos por clase solo donde pasaron la regla (k = 12)
    proba_clf[k], pred_clf[k] = prw / prw.sum(1, keepdims=True), prw.argmax(1)

# comprobación: deben coincidir con las predicciones reportadas en el paper
ref = pd.read_csv(os.path.join(FINAL, "resultados", "test_predicciones.csv"))
for k, c in ((0, "F0"), (4, "F1"), (8, "F1"), (12, "F1")):
    r = ref[(ref.tarea == "reg") & (ref.modelo == "xgb") & (ref.k == k) & (ref.conjunto == c)].pred.values
    assert np.allclose(r, pred_reg[k], atol=1e-4), f"regresión k={k} no coincide con el paper"
    mo = "xgb+pesos" if k == 12 else "xgb"
    r = ref[(ref.tarea == "clf") & (ref.umbral == "cao") & (ref.modelo == mo) & (ref.k == k) & (ref.conjunto == c)].pred.values
    assert (r == pred_clf[k]).all(), f"clasificación k={k} no coincide con el paper"
print("Predicciones idénticas a las del paper ✔")

import shap  # noqa: E402
a0 = MOD("xgb_reg_k0_F0.joblib")
sv = shap.TreeExplainer(a0["modelo"])(X[0][a0["columnas"]].iloc[IDX_TEST])
base_shap = float(np.atleast_1d(sv.base_values)[0])

vueltas = []
for j, i in enumerate(IDX_TEST):
    h = vm.hora_salida.iat[i]
    vueltas.append({
        "id": int(vm.id.iat[i]), "fecha": vm.date.iat[i], "hora": round(float(h), 4), "dia": int(vm.dia_semana.iat[i]),
        "veh": vm.vehicle_id_id.iat[i], "vuelta": int(vm.lap.iat[i]), "horario": int(vm.dur_prog.iat[i]),
        "d": [r2(v) for v in Dm[i]], "y": r2(vm.y.iat[i]), "clase": int(a_clase([vm.y.iat[i]])[0]),
        "pred": {str(k): r2(pred_reg[k][j]) for k in pred_reg},
        "pclase": {str(k): int(pred_clf[k][j]) for k in pred_clf},
        "proba": {str(k): [round(float(p), 3) for p in proba_clf[k][j]] for k in proba_clf},
        "shap": [round(float(v), 2) for v in sv.values[j]],
    })

# ---------------------------------------------------------------- simulador "antes de salir" (k = 0)
# Para cada (día, hora, horario) se promedia la predicción sobre los 45 vehículos de la flota, con el n.º de vuelta
# más frecuente a esa hora. Es un promedio del modelo, no una vuelta real; se indica si la combinación no se observó.
ar, ac = MOD("xgb_reg_k0_F0.joblib"), MOD("xgb_clf_k0_F0.joblib")
vehs = sorted(vm.vehicle_id_id.astype(int).unique())
horas = np.round(np.arange(4.0, 20.01, 0.25), 2)
lap_hora = vm.assign(hb=np.floor(vm.hora_salida)).groupby("hb").lap.agg(lambda s: int(s.mode().iat[0]))
HORARIOS = [147, 161, 168, 170]
filas = []
for dia in range(7):
    for h in horas:
        for hor in HORARIOS:
            lap = int(lap_hora.get(np.floor(h), lap_hora.iloc[(lap_hora.index - np.floor(h)).__abs__().argmin()]))
            for v in vehs:
                filas.append((h, dia, lap, hor, v))
G = pd.DataFrame(filas, columns=["hora_salida", "dia_semana", "lap", "dur_prog", "vehiculo"])
G["pred"] = ar["modelo"].predict(G[ar["columnas"]])
P = ac["modelo"].predict_proba(G[ac["columnas"]])
for c in range(3):
    G[f"p{c}"] = P[:, c]
sim = G.groupby(["dia_semana", "hora_salida", "dur_prog"])[["pred", "p0", "p1", "p2"]].mean().reset_index()
obs = vm.assign(hb=np.floor(vm.hora_salida)).groupby(["dia_semana", "hb", "dur_prog"]).size()
simulador = {}
for r in sim.itertuples():
    n = int(obs.get((r.dia_semana, np.floor(r.hora_salida), r.dur_prog), 0))
    simulador[f"{r.dia_semana}|{r.hora_salida:.2f}|{int(r.dur_prog)}"] = [round(r.pred, 2), round(r.p0, 3), round(r.p1, 3), round(r.p2, 3), n]
tmp = vm.assign(hb=np.floor(vm.hora_salida))
horario_frec = {f"{int(hb)}": int(g.dur_prog.mode().iat[0]) for hb, g in tmp.groupby("hb")}                 # respaldo
horario_frec.update({f"{int(dd)}|{int(hb)}": int(g.dur_prog.mode().iat[0]) for (dd, hb), g in tmp.groupby(["dia_semana", "hb"])})

# ---------------------------------------------------------------- resultados (desde los CSV del paper)
R = lambda f: pd.read_csv(os.path.join(FINAL, "resultados", f))
res = R("test_resultados.csv")
mae_k = {}
for mo in ["B2 histórico", "B1 persistencia", "ridge", "rf", "xgb"]:
    q = res[(res.tarea == "reg") & (res.modelo == mo) & (res.conjunto != "F2")].sort_values("k")
    mae_k[mo] = [[int(k), round(v, 3)] for k, v in zip(q.k, q.MAE)]
f1_k = {}
for mo in ["B2 histórico", "B1 persistencia", "logistic", "rf", "xgb"]:
    q = res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == mo) & (res.conjunto != "F2")].sort_values("k")
    f1_k[mo] = [[int(k), round(v, 3)] for k, v in zip(q.k, q.F1_macro)]
dia = R("test_error_por_dia.csv")
por_dia = [{"fecha": r.date, "vueltas": int(r.vueltas), "retraso": round(r.retraso_medio, 2), "mae": round(r.MAE, 2),
            "pct": round(r.pct_error, 1)} for r in dia.itertuples()]
pr = R("test_predicciones.csv")
conf = {}
for nom, mo, k, c in (("k0", "xgb", 0, "F0"), ("k12", "xgb+pesos", 12, "F1")):
    q = pr[(pr.tarea == "clf") & (pr.umbral == "cao") & (pr.modelo == mo) & (pr.k == k) & (pr.conjunto == c)]
    conf[nom] = pd.crosstab(q.real, q.pred).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values.tolist()
imp = R("importancia_permutacion.csv")
imp0 = imp[(imp.tarea == "reg") & (imp.k == 0)].sort_values("media", ascending=False)
importancia = [[v, round(m, 3)] for v, m in zip(imp0.variable, imp0.media)]
sens = res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "xgb") & (res.k == 0)].iloc[0]

DATOS = {"pc": [{"lat": a, "lon": b, "desc": c} for a, b, c in PC], "traza": traza, "perfil": perfil,
         "vueltas": vueltas, "base_shap": round(base_shap, 3), "shap_vars": list(a0["columnas"]),
         "simulador": simulador, "horas": horas.tolist(), "horario_frec": horario_frec,
         "mae_k": mae_k, "f1_k": f1_k, "por_dia": por_dia, "conf": conf, "importancia": importancia,
         "metricas_k0": {"MAE": round(float(res[(res.tarea == "reg") & (res.modelo == "xgb") & (res.k == 0)].MAE.iat[0]), 2),
                         "MAE_B2": round(float(res[(res.tarea == "reg") & (res.modelo == "B2 histórico") & (res.k == 0)].MAE.iat[0]), 2),
                         "R2": round(float(res[(res.tarea == "reg") & (res.modelo == "xgb") & (res.k == 0)].R2.iat[0]), 2),
                         "F1": round(float(sens.F1_macro), 3), "exactitud": round(float(sens.exactitud), 3),
                         "kappa": round(float(sens.kappa), 4)}}

with open(os.path.join(AQUI, "plantilla.html"), encoding="utf8") as fh:
    html = fh.read()
html = html.replace("/*__DATOS__*/null", json.dumps(DATOS, ensure_ascii=False, separators=(",", ":")))
with open(os.path.join(AQUI, "index.html"), "w", encoding="utf8") as fh:
    fh.write(html)
print(f"Demo escrita: Demo/index.html ({os.path.getsize(os.path.join(AQUI, 'index.html')) / 1024:.0f} KB)")
