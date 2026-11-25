"""Variables, modelos y métricas. Idéntico al notebook del Entregable 2 (etapas 5, 7 y 8), con dos extensiones:
  - construir_features acepta grupos extra (C: contexto del día, P: planificación).
  - metricas_clf agrega exactitud, exactitud balanceada y sensibilidad/especificidad por clase (uno contra el resto).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, f1_score, recall_score,
                             accuracy_score, balanced_accuracy_score, confusion_matrix, average_precision_score)

SEED = 42
CAT_COLS = ["dia_semana", "lap", "dur_prog", "vehiculo"]
MODO_X = {"xgb": {}, "rf": {"imputar": True}, "ridge": {"para_lineal": True}, "logistic": {"para_lineal": True}}


def construir_features(vm, Dm, k, conjunto="F1", gps=None, para_lineal=False, imputar=None, extra=None):
    """Igual que en E2. `extra`: lista de DataFrames (grupos C/P) alineados con vm que se agregan al final.
    En RF y lineales, los faltantes de los grupos extra se imputan con la mediana del entrenamiento (dentro del
    Pipeline) y se agrega un indicador de faltante; XGBoost usa NaN de forma nativa."""
    if imputar is None:
        imputar = para_lineal
    X = pd.DataFrame({"hora_salida": vm["hora_salida"].values, "dia_semana": vm["dia_semana"].values,
                      "lap": vm["lap"].values, "dur_prog": vm["dur_prog"].values,
                      "vehiculo": vm["vehicle_id_id"].astype(int).values})
    if para_lineal:
        X["hora_salida2"] = X["hora_salida"] ** 2
        for c in CAT_COLS:
            X[c] = X[c].astype(str)
    if conjunto in ("F1", "F2") and k > 0:
        Dk = pd.DataFrame(Dm[:, :k], columns=[f"d{j}" for j in range(1, k + 1)])
        if imputar:
            ind = Dk.isna().astype(int).add_suffix("_falta")
            Dk = pd.concat([Dk.ffill(axis=1).fillna(0), ind], axis=1)
        X = pd.concat([X, Dk], axis=1)
    if conjunto == "F2" and k > 0:
        X = pd.concat([X, gps.reset_index(drop=True)], axis=1)
    for g in (extra or []):
        g = g.reset_index(drop=True)
        X = pd.concat([X, g], axis=1)
        if imputar:
            falt = [c for c in g.columns if g[c].isna().any()]
            if falt:
                X = pd.concat([X, g[falt].isna().astype(int).add_suffix("_falta")], axis=1)
    return X


def preprocesador_lineal(X):
    cat = [c for c in CAT_COLS if c in X.columns]
    ind = [c for c in X.columns if c.endswith("_falta")]
    num = [c for c in X.columns if c not in cat and c not in ind]
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("ind", "passthrough", ind)])


def crear_modelo(tarea, modelo, params, X, n_jobs=-1):
    p = dict(params)
    if modelo == "ridge":
        return Pipeline([("pre", preprocesador_lineal(X)), ("m", Ridge(**p))])
    if modelo == "logistic":
        return Pipeline([("pre", preprocesador_lineal(X)), ("m", LogisticRegression(max_iter=3000, **p))])
    if modelo == "rf":
        cls = RandomForestRegressor if tarea == "reg" else RandomForestClassifier
        return Pipeline([("imp", SimpleImputer(strategy="median")), ("m", cls(random_state=SEED, n_jobs=n_jobs, **p))])
    if modelo == "xgb":
        p.pop("class_weight", None)
        cls = XGBRegressor if tarea == "reg" else XGBClassifier
        return cls(random_state=SEED, n_jobs=n_jobs, tree_method="hist", **p)
    raise ValueError(modelo)


def ajustar(tarea, modelo, params, X, y, n_jobs=-1):
    m = crear_modelo(tarea, modelo, params, X, n_jobs)
    kw = {}
    if modelo == "xgb" and tarea == "clf" and params.get("class_weight") == "balanced":
        kw["sample_weight"] = compute_sample_weight("balanced", y)
    return m.fit(X, y, **kw)


def predecir(tarea, m, X):
    if tarea == "reg":
        return m.predict(X), None
    proba = m.predict_proba(X)
    return proba.argmax(1), proba


def predecir_variantes(tarea, modelo, m, X, n_list):
    if n_list == [None]:
        return {None: predecir(tarea, m, X)}
    res = {}
    for n in n_list:
        if modelo == "xgb":
            if tarea == "reg":
                res[n] = (m.predict(X, iteration_range=(0, n)), None)
            else:
                pr = m.predict_proba(X, iteration_range=(0, n))
                res[n] = (pr.argmax(1), pr)
        else:
            Xt, est = m[:-1].transform(X), m[-1].estimators_[:n]
            if tarea == "reg":
                res[n] = (np.mean([t.predict(Xt) for t in est], axis=0), None)
            else:
                pr = np.mean([t.predict_proba(Xt) for t in est], axis=0)
                res[n] = (pr.argmax(1), pr)
    return res


def metricas_reg(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p)}


def metricas_clf(y, p, proba=None):
    m = {"F1_macro": f1_score(y, p, average="macro"),
         "recall_retrasado": recall_score(y, p, labels=[2], average="macro", zero_division=0),
         "exactitud": accuracy_score(y, p), "exactitud_bal": balanced_accuracy_score(y, p)}
    cm = confusion_matrix(y, p, labels=[0, 1, 2])
    for c, nom in enumerate(["adel", "punt", "retr"]):
        tp = cm[c, c]; fn = cm[c].sum() - tp; fp = cm[:, c].sum() - tp; tn = cm.sum() - tp - fn - fp
        m[f"sens_{nom}"] = tp / (tp + fn) if tp + fn else np.nan
        m[f"espec_{nom}"] = tn / (tn + fp) if tn + fp else np.nan
    if proba is not None:
        m["PR_AUC"] = average_precision_score(label_binarize(y, classes=[0, 1, 2]), proba, average="macro")
    return m
