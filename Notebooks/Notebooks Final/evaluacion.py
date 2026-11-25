"""Evaluación de la entrega final: selección por validación, evaluación ÚNICA en prueba, pruebas estadísticas,
regla de decisión por clase (D4b), XGBoost vs. Random Forest (D6), importancia y SHAP.

Lee resultados/cv_raw.csv (experimento.py) y escribe tablas en resultados/ y modelos en modelos/.
"""
import json, os, pickle, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix
from sklearn.inspection import permutation_importance
from prep import cargar, a_clase
from modelos import construir_features, ajustar, predecir, metricas_reg, metricas_clf, MODO_X, SEED
from experimento import ESCENARIOS, UMBRALES

warnings.filterwarnings("ignore")
AQUI = os.path.dirname(os.path.abspath(__file__))
OUT, MOD = os.path.join(AQUI, "resultados"), os.path.join(AQUI, "modelos")
os.makedirs(MOD, exist_ok=True)
ALPHA = 0.05
MODELOS = {"reg": ["ridge", "rf", "xgb"], "clf": ["logistic", "rf", "xgb"]}

d = cargar()
vm, Dm, FOLDS, IDX_DEV, IDX_TEST = d["vm"], d["Dm"], d["FOLDS"], d["IDX_DEV"], d["IDX_TEST"]
y_reg = vm["y"].values
Y = {("reg", "-"): y_reg}
for u, (a, b) in UMBRALES.items():
    Y[("clf", u)] = a_clase(y_reg, a, b).astype(int)

XS = {}
for k, conj in ESCENARIOS:
    gps = d["GPS_K"][k] if conj == "F2" else None
    XS[(k, conj)] = {m: construir_features(vm, Dm, k, conj, gps, **kw) for m, kw in MODO_X.items()}

# ------------------------------------------------------------------ selección por validación
cv = pd.read_csv(os.path.join(OUT, "cv_raw.csv"))
cv["orden"] = np.where(cv.tarea == "reg", cv.metrica_media, -cv.metrica_media)
G = ["tarea", "umbral", "modelo", "k", "conjunto"]
mejores = cv.sort_values(["orden", "clave"]).drop_duplicates(G).drop(columns="orden").reset_index(drop=True)
mejores_orig = (cv[cv.malla_original].sort_values(["orden", "clave"]).drop_duplicates(G).drop(columns="orden")
                .reset_index(drop=True))
mejores.to_csv(os.path.join(OUT, "cv_mejores.csv"), index=False)
mejores_orig.to_csv(os.path.join(OUT, "cv_mejores_malla_original.csv"), index=False)


def mejor_modelo(tarea, umbral, k, conj):
    c = mejores[(mejores.tarea == tarea) & (mejores.umbral == umbral) & (mejores.k == k) & (mejores.conjunto == conj)]
    return c.sort_values("metrica_media", ascending=(tarea == "reg")).modelo.iat[0]


# ------------------------------------------------------------------ líneas base
def pred_persistencia(D_, k):
    return pd.DataFrame(D_[:, :k]).ffill(axis=1).iloc[:, -1].fillna(0).values


def pred_historico(idx_tr, idx_ap):
    h_tr, h_ap = np.floor(vm.hora_salida.values[idx_tr]), np.floor(vm.hora_salida.values[idx_ap])
    medias = pd.Series(y_reg[idx_tr]).groupby(h_tr).mean()
    return pd.Series(h_ap).map(medias).fillna(y_reg[idx_tr].mean()).values


def lineas_base(idx_tr, idx_ap, k, umbral):
    a, b = UMBRALES[umbral]
    yc = Y[("clf", umbral)]
    may = np.bincount(yc[idx_tr]).argmax()
    out = {"B2 histórico": {"reg": pred_historico(idx_tr, idx_ap), "clf": np.full(len(idx_ap), may),
                            "proba": np.eye(3)[np.full(len(idx_ap), may)]}}
    if k > 0:
        p = pred_persistencia(Dm[idx_ap], k)
        c = a_clase(p, a, b).astype(int)
        out["B1 persistencia"] = {"reg": p, "clf": c, "proba": np.eye(3)[c]}
    return out


filas = []
for u in UMBRALES:
    for k in (0, 4, 8, 12):
        for n, (a, b) in enumerate(FOLDS, 1):
            for nombre, pr in lineas_base(a, b, k, u).items():
                if u == "cao":
                    filas.append({"tarea": "reg", "umbral": "-", "modelo": nombre, "k": k, "pliegue": n,
                                  "valor": metricas_reg(y_reg[b], pr["reg"])["MAE"]})
                filas.append({"tarea": "clf", "umbral": u, "modelo": nombre, "k": k, "pliegue": n,
                              "valor": metricas_clf(Y[("clf", u)][b], pr["clf"])["F1_macro"]})
cv_base = pd.DataFrame(filas).groupby(["tarea", "umbral", "modelo", "k"]).valor.agg(["mean", "std"]).reset_index()
cv_base.to_csv(os.path.join(OUT, "cv_lineas_base.csv"), index=False)

# ------------------------------------------------------------------ entrenamiento final y evaluación única en prueba
PRED, FINALES = {}, {}
for r in mejores.itertuples():
    X, y = XS[(r.k, r.conjunto)][r.modelo], Y[(r.tarea, r.umbral)]
    m = ajustar(r.tarea, r.modelo, json.loads(r.params), X.iloc[IDX_DEV], y[IDX_DEV], n_jobs=1)
    FINALES[(r.tarea, r.umbral, r.modelo, r.k, r.conjunto)] = m
    PRED[(r.tarea, r.umbral, r.modelo, r.k, r.conjunto)] = predecir(r.tarea, m, X.iloc[IDX_TEST])
for u in UMBRALES:
    for k in (0, 4, 8, 12):
        for nombre, pr in lineas_base(IDX_DEV, IDX_TEST, k, u).items():
            for conj in (["F0"] if k == 0 else ["F1", "F2"]):
                if u == "cao":
                    PRED[("reg", "-", nombre, k, conj)] = (pr["reg"], None)
                PRED[("clf", u, nombre, k, conj)] = (pr["clf"], pr["proba"])

YT = {key: y[IDX_TEST] for key, y in Y.items()}
filas = []
for (t, u, mo, k, c), (p, pr) in PRED.items():
    yt = YT[(t, u)]
    met = metricas_reg(yt, p) if t == "reg" else {**metricas_clf(yt, p, pr), "kappa": cohen_kappa_score(yt, p)}
    filas.append({"tarea": t, "umbral": u, "modelo": mo, "k": k, "conjunto": c, **met})
res_test = pd.DataFrame(filas)
res_test.to_csv(os.path.join(OUT, "test_resultados.csv"), index=False)
RT = res_test.set_index(["tarea", "umbral", "modelo", "k", "conjunto"])


# ------------------------------------------------------------------ pruebas pareadas
def p_wilcoxon(ea, eb):
    return wilcoxon(ea, eb).pvalue if np.any(ea != eb) else 1.0


def p_mcnemar(oa, ob):
    b, c = int((oa & ~ob).sum()), int((~oa & ob).sum())
    tabla = [[int((oa & ob).sum()), b], [c, int((~oa & ~ob).sum())]]
    ex = (b + c) < 25
    return mcnemar(tabla, exact=ex, correction=not ex).pvalue


def comparar(ka, kb, pred=None):
    pred = pred or PRED
    t, u = ka[0], ka[1]
    pa, pb, yt = pred[ka][0], pred[kb][0], YT[(t, u)]
    return p_wilcoxon(np.abs(yt - pa), np.abs(yt - pb)) if t == "reg" else p_mcnemar(pa == yt, pb == yt)


filas = []  # H1
for t, u in (("reg", "-"), ("clf", "cao"), ("clf", "amplio")):
    met = "MAE" if t == "reg" else "F1_macro"
    for k, c in ESCENARIOS:
        best = mejor_modelo(t, u, k, c)
        for base in (["B2 histórico"] if k == 0 else ["B1 persistencia", "B2 histórico"]):
            filas.append({"tarea": t, "umbral": u, "k": k, "conjunto": c, "mejor": best, "valor_mejor": RT.loc[(t, u, best, k, c), met],
                          "base": base, "valor_base": RT.loc[(t, u, base, k, c), met],
                          "p": comparar((t, u, best, k, c), (t, u, base, k, c))})
h1 = pd.DataFrame(filas)
h1.to_csv(os.path.join(OUT, "test_h1.csv"), index=False)

filas = []  # H2
for t, u in (("reg", "-"), ("clf", "cao")):
    met = "MAE" if t == "reg" else "F1_macro"
    for mo in MODELOS[t]:
        pasos = [(0, "F0"), (4, "F1"), (8, "F1"), (12, "F1")]
        for (k1, c1), (k2, c2) in list(zip(pasos[:-1], pasos[1:])) + [((0, "F0"), (12, "F1"))]:
            filas.append({"tarea": t, "modelo": mo, "de": k1, "a": k2, "valor_de": RT.loc[(t, u, mo, k1, c1), met],
                          "valor_a": RT.loc[(t, u, mo, k2, c2), met], "p": comparar((t, u, mo, k1, c1), (t, u, mo, k2, c2))})
h2 = pd.DataFrame(filas)
h2.to_csv(os.path.join(OUT, "test_h2.csv"), index=False)

filas = []  # H3
for t, u in (("reg", "-"), ("clf", "cao")):
    met = "MAE" if t == "reg" else "F1_macro"
    for mo in MODELOS[t]:
        for k in (4, 8, 12):
            filas.append({"tarea": t, "modelo": mo, "k": k, "valor_F1": RT.loc[(t, u, mo, k, "F1"), met],
                          "valor_F2": RT.loc[(t, u, mo, k, "F2"), met], "p": comparar((t, u, mo, k, "F1"), (t, u, mo, k, "F2"))})
h3 = pd.DataFrame(filas)
h3.to_csv(os.path.join(OUT, "test_h3.csv"), index=False)

filas = []  # D6: XGBoost vs Random Forest, Holm por tarea (7 escenarios)
for t, u in (("reg", "-"), ("clf", "cao")):
    met = "MAE" if t == "reg" else "F1_macro"
    ps = []
    for k, c in ESCENARIOS:
        p = comparar((t, u, "xgb", k, c), (t, u, "rf", k, c))
        ps.append(p)
        filas.append({"tarea": t, "k": k, "conjunto": c, "XGB": RT.loc[(t, u, "xgb", k, c), met],
                      "RF": RT.loc[(t, u, "rf", k, c), met], "p": p})
    rech, p_holm, _, _ = multipletests(ps, alpha=ALPHA, method="holm")
    for i, (ph, rj) in enumerate(zip(p_holm, rech)):
        filas[-7 + i]["p_holm"], filas[-7 + i]["significativo_holm"] = ph, rj
d6 = pd.DataFrame(filas)
d6.to_csv(os.path.join(OUT, "test_xgb_vs_rf.csv"), index=False)

# ------------------------------------------------------------------ D4b: pesos por clase (validación anidada)
PESOS = np.round(np.arange(0.5, 3.01, 0.1), 2)


def mejores_pesos(proba, y):
    best, bw = -1, (1.0, 1.0)
    for w0 in PESOS:
        for w2 in PESOS:
            f = f1_score(y, (proba * np.array([w0, 1.0, w2])).argmax(1), average="macro")
            if f > best + 1e-12:
                best, bw = f, (w0, w2)
    return bw


filas_d4b, PESOS_FIN = [], {}
for k, c in ESCENARIOS:
    best = mejor_modelo("clf", "cao", k, c)
    params = json.loads(mejores[(mejores.tarea == "clf") & (mejores.umbral == "cao") & (mejores.modelo == best)
                                & (mejores.k == k) & (mejores.conjunto == c)].params.iat[0])
    X, y = XS[(k, c)][best], Y[("clf", "cao")]
    oof = []
    for a, b in FOLDS:
        m = ajustar("clf", best, params, X.iloc[a], y[a], n_jobs=1)
        oof.append((b, m.predict_proba(X.iloc[b])))
    f_sin, f_con, pesos_j = [], [], []
    for j in range(4):  # pesos elegidos con los otros 3 pliegues, evaluados en el pliegue j
        pr_o = np.vstack([oof[i][1] for i in range(4) if i != j]); y_o = np.concatenate([y[oof[i][0]] for i in range(4) if i != j])
        w0, w2 = mejores_pesos(pr_o, y_o)
        pesos_j.append((w0, w2))
        b, pr = oof[j]
        f_sin.append(f1_score(y[b], pr.argmax(1), average="macro"))
        f_con.append(f1_score(y[b], (pr * np.array([w0, 1, w2])).argmax(1), average="macro"))
    w0, w2 = mejores_pesos(np.vstack([o[1] for o in oof]), np.concatenate([y[o[0]] for o in oof]))
    ganados = int(np.sum(np.array(f_con) > np.array(f_sin)))
    acepta = (np.mean(f_con) > np.mean(f_sin)) and ganados >= 3
    PESOS_FIN[(k, c)] = (w0, w2, acepta, best)
    filas_d4b.append({"k": k, "conjunto": c, "modelo": best, "F1_val_sin": np.mean(f_sin), "F1_val_con": np.mean(f_con),
                      "pliegues_ganados": ganados, "acepta_D1": acepta, "w_adelantado": w0, "w_retrasado": w2,
                      "pesos_por_pliegue": str(pesos_j)})
d4b = pd.DataFrame(filas_d4b)

# Aplicación en prueba (se reporta siempre; se ADOPTA solo si pasó D1 en validación)
PRED_W = dict(PRED)
for (k, c), (w0, w2, acepta, best) in PESOS_FIN.items():
    _, pr = PRED[("clf", "cao", best, k, c)]
    PRED_W[("clf", "cao", best + "+pesos", k, c)] = ((pr * np.array([w0, 1, w2])).argmax(1), pr)
filas = []
for (k, c), (w0, w2, acepta, best) in PESOS_FIN.items():
    yt = YT[("clf", "cao")]
    for nom in (best, best + "+pesos"):
        p, pr = PRED_W[("clf", "cao", nom, k, c)]
        filas.append({"k": k, "conjunto": c, "modelo": nom, **metricas_clf(yt, p, pr), "kappa": cohen_kappa_score(yt, p)})
    d4b.loc[(d4b.k == k) & (d4b.conjunto == c), "p_mcnemar_test"] = comparar(("clf", "cao", best + "+pesos", k, c),
                                                                             ("clf", "cao", best, k, c), PRED_W)
d4b.to_csv(os.path.join(OUT, "d4b_pesos_validacion.csv"), index=False)
pd.DataFrame(filas).to_csv(os.path.join(OUT, "d4b_pesos_test.csv"), index=False)

# ------------------------------------------------------------------ error por día (mejor regresor, k = 0)
best0 = mejor_modelo("reg", "-", 0, "F0")
p0 = PRED[("reg", "-", best0, 0, "F0")][0]
t = vm.iloc[IDX_TEST].assign(pred=p0, err=lambda x: (x.y - x.pred).abs())
por_dia = t.groupby("date").agg(vueltas=("y", "size"), retraso_medio=("y", "mean"), MAE=("err", "mean"), err_total=("err", "sum"))
por_dia["pct_error"] = 100 * por_dia.err_total / por_dia.err_total.sum()
por_dia.to_csv(os.path.join(OUT, "test_error_por_dia.csv"))

# ------------------------------------------------------------------ importancia por permutación (como E2)
imps = []
for t_, u, k, c in (("reg", "-", 0, "F0"), ("reg", "-", 12, "F2"), ("clf", "cao", 12, "F2"), ("reg", "-", 12, "F1")):
    mo = mejor_modelo(t_, u, k, c)
    X = XS[(k, c)][mo].iloc[IDX_TEST]
    pi = permutation_importance(FINALES[(t_, u, mo, k, c)], X, YT[(t_, u)], n_repeats=10, random_state=SEED, n_jobs=1,
                                scoring="neg_mean_absolute_error" if t_ == "reg" else "f1_macro")
    imps.append(pd.DataFrame({"variable": X.columns, "media": pi.importances_mean, "std": pi.importances_std,
                              "tarea": t_, "modelo": mo, "k": k, "conjunto": c}))
pd.concat(imps).to_csv(os.path.join(OUT, "importancia_permutacion.csv"), index=False)

# ------------------------------------------------------------------ predicciones y modelos guardados
largo = []
for (t_, u, mo, k, c), (p, pr) in PRED_W.items():
    largo.append(pd.DataFrame({"id": vm.id.values[IDX_TEST], "date": vm.date.values[IDX_TEST], "tarea": t_, "umbral": u,
                               "modelo": mo, "k": k, "conjunto": c, "real": YT[(t_, u)], "pred": p}))
pd.concat(largo).to_csv(os.path.join(OUT, "test_predicciones.csv"), index=False)

for (t_, u, mo, k, c), m in FINALES.items():
    if mo == "xgb" and c in ("F0", "F1") and u in ("-", "cao"):
        joblib.dump({"modelo": m, "columnas": list(XS[(k, c)]["xgb"].columns), "tarea": t_, "k": k, "conjunto": c,
                     "pesos_clase": PESOS_FIN.get((k, c)) if t_ == "clf" else None},
                    os.path.join(MOD, f"xgb_{t_}_k{k}_{c}.joblib"))
with open(os.path.join(MOD, "estado_eval.pkl"), "wb") as fh:
    pickle.dump({"PRED": PRED_W, "YT": YT, "IDX_TEST": IDX_TEST}, fh)
print("Evaluación terminada")
