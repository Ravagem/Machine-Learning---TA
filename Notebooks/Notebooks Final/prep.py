"""Preparación de datos de la entrega final.

Reproduce sin cambios las etapas 3 (preprocesamiento), 5 (variables F0/F1/F2) y 6 (partición temporal)
del notebook del Entregable 2, y agrega los grupos de variables nuevos de la entrega final:

  C (contexto del día en curso): retraso de las vueltas del mismo día ya terminadas antes de la salida.
  P (planificación): frecuencia programada alrededor de la salida (conocida de antemano).

Uso:  from prep import cargar;  d = cargar()   (cachea en prep_cache.pkl)
"""
import os, pickle, time
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(AQUI, "..", "..", "Dataset"))
CACHE = os.path.join(AQUI, "prep_cache.pkl")

N_PC, K_VALUES, MAX_ABS_DEV = 18, [0, 4, 8, 12], 60
UMBRAL_ADELANTO, UMBRAL_RETRASO = -1, 5
TEST_INICIO, DEV_INICIO = "2025-01-28", "2025-01-06"
CV_FOLDS = [("2025-01-15", "2025-01-16", "2025-01-18"), ("2025-01-18", "2025-01-19", "2025-01-21"),
            ("2025-01-21", "2025-01-22", "2025-01-24"), ("2025-01-24", "2025-01-25", "2025-01-27")]
VEL_DETENIDO, DT_MAX = 3, 300


def a_clase(y, u_adel=UMBRAL_ADELANTO, u_retr=UMBRAL_RETRASO):
    y = np.asarray(y, dtype=float)
    c = np.where(y < u_adel, 0, np.where(y <= u_retr, 1, 2)).astype(float)
    c[np.isnan(y)] = np.nan
    return c


def _parse(c):
    return c.strip().strip("{}").split(",")


def _seg(h):
    a, b, s = map(int, h.strip().split(":"))
    return a * 3600 + b * 60 + s


def _agregados(ss, sp, t0, t1):
    if np.isnan(t0) or np.isnan(t1) or t1 <= t0:
        return [np.nan] * 3
    a, b = np.searchsorted(ss, t0), np.searchsorted(ss, t1, side="right")
    if b - a < 3:
        return [np.nan] * 3
    s2, v2 = ss[a:b], sp[a:b]
    dt = np.clip(np.diff(s2, append=t1), 0, DT_MAX)
    if dt.sum() <= 0:
        return [np.nan] * 3
    vmed = np.average(v2, weights=dt)
    vdesv = np.sqrt(np.average((v2 - vmed) ** 2, weights=dt))
    return [vmed, vdesv, dt[v2 <= VEL_DETENIDO].sum() / 60]


def contexto_dia(vueltas, E_ok, S, n_ult=5):
    """Grupo C. Para cada vuelta i, con corte en su hora PROGRAMADA de salida S[i,0] (anti-fuga):
    solo cuentan las vueltas del mismo día cuya llegada REAL a PC18 es anterior al corte.
      dia_retraso_medio : media del retraso final de esas vueltas (NaN si aún no terminó ninguna)
      dia_retraso_ult5  : media de las últimas n_ult en terminar (tendencia reciente del día)
      dia_n_terminadas  : cuántas vueltas del día ya terminaron
      veh_retraso_prev  : retraso final de la vuelta anterior del mismo vehículo, si ya terminó
    Las vueltas sin retraso final válido no aportan (su llegada real a PC18 es inválida)."""
    y = vueltas["y"].values
    fin = E_ok[:, 17]                        # llegada real a PC18 (NaN si inválida)
    corte = S[:, 0]
    out = np.full((len(vueltas), 4), np.nan)
    for fecha, idx in vueltas.groupby("date").indices.items():
        idx = np.asarray(idx)
        val = idx[~np.isnan(fin[idx])]
        orden = val[np.argsort(fin[val], kind="stable")]
        f_ord, y_ord = fin[orden], y[orden]
        veh = vueltas["vehicle_id_id"].values
        for i in idx:
            n = np.searchsorted(f_ord, corte[i], side="left")      # terminadas estrictamente antes del corte
            out[i, 2] = n
            if n > 0:
                out[i, 0] = y_ord[:n].mean()
                out[i, 1] = y_ord[max(0, n - n_ult):n].mean()
                prev = [j for j in orden[:n] if veh[j] == veh[i]]
                if prev:
                    out[i, 3] = y[prev[-1]]
    return pd.DataFrame(out, columns=["dia_retraso_medio", "dia_retraso_ult5", "dia_n_terminadas", "veh_retraso_prev"])


def anomalia_dia(vueltas, E_ok, S, n_ult=5):
    """Grupo A (segundo intento, declarado tras rechazar C). C mezcla el estado del día con la hora: antes de las 8 h
    solo han terminado vueltas de madrugada, que van adelantadas. A mide cuánto PEOR DE LO HABITUAL va el día:
      residuo_j = y_j − media histórica de y a la hora de salida de j, calculada solo con DÍAS ANTERIORES (sin fuga).
      dia_anom_media, dia_anom_ult5 : media de los residuos de las vueltas del día terminadas antes de la salida
      veh_anom_prev                 : residuo de la vuelta anterior del mismo vehículo, si ya terminó"""
    y = vueltas["y"].values
    hora = np.floor(vueltas["hora_salida"].values)
    fechas = vueltas["date"].values
    resid = np.full(len(vueltas), np.nan)
    for f in np.unique(fechas):
        hoy, prev = fechas == f, (fechas < f) & ~np.isnan(y)
        if prev.sum() == 0:
            continue
        medias = pd.Series(y[prev]).groupby(hora[prev]).mean()
        glob = y[prev].mean()
        resid[hoy] = y[hoy] - pd.Series(hora[hoy]).map(medias).fillna(glob).values
    fin, corte, veh = E_ok[:, 17], S[:, 0], vueltas["vehicle_id_id"].values
    out = np.full((len(vueltas), 3), np.nan)
    for fecha, idx in vueltas.groupby("date").indices.items():
        idx = np.asarray(idx)
        val = idx[~np.isnan(fin[idx]) & ~np.isnan(resid[idx])]
        orden = val[np.argsort(fin[val], kind="stable")]
        f_ord, r_ord = fin[orden], resid[orden]
        for i in idx:
            n = np.searchsorted(f_ord, corte[i], side="left")
            if n > 0:
                out[i, 0] = r_ord[:n].mean()
                out[i, 1] = r_ord[max(0, n - n_ult):n].mean()
                prev = [j for j in orden[:n] if veh[j] == veh[i]]
                if prev:
                    out[i, 2] = resid[prev[-1]]
    return pd.DataFrame(out, columns=["dia_anom_media", "dia_anom_ult5", "veh_anom_prev"])


def planificacion(vueltas_todas_S0, fechas_todas, vueltas):
    """Grupo P. A partir del itinerario (horas programadas de salida de TODAS las vueltas del día,
    conocidas de antemano):
      prog_salidas_60 : n.º de salidas programadas en los 60 min previos a la salida (sin contarla)
      prog_headway    : minutos desde la salida programada anterior (cualquier vehículo)"""
    out = np.full((len(vueltas), 2), np.nan)
    por_fecha = {f: np.sort(vueltas_todas_S0[fechas_todas == f]) for f in np.unique(fechas_todas)}
    for i, (f, s0) in enumerate(zip(vueltas["date"].values, vueltas["salida_prog_seg"].values)):
        arr = por_fecha[f]
        a, b = np.searchsorted(arr, s0 - 3600, side="left"), np.searchsorted(arr, s0, side="left")
        out[i, 0] = b - a
        out[i, 1] = (s0 - arr[b - 1]) / 60 if b > 0 else np.nan
    return pd.DataFrame(out, columns=["prog_salidas_60", "prog_headway"])


def _preparar():
    t0 = time.time()
    times_raw = pd.read_csv(os.path.join(DATA_DIR, "Times.csv"), dtype={"vehicle_id_id": str})
    mon_raw = pd.read_csv(os.path.join(DATA_DIR, "Monitoring.csv"), dtype={"vehicle_id_id": str})
    assert times_raw.shape == (4707, 8) and mon_raw.shape == (1630431, 9)

    S = np.array([[_seg(x) for x in _parse(c)] for c in times_raw.start_time], dtype=float)
    E = np.array([[_seg(x) for x in _parse(c)] for c in times_raw.end_time], dtype=float)
    D = np.array([[np.nan if x.strip() == "-" else float(x) for x in _parse(c)] for c in times_raw.difference_time], dtype=float)
    assert int(np.isnan(D).sum()) == 7746
    D[np.abs(D) > MAX_ABS_DEV] = np.nan
    monotono = (np.diff(S, axis=1) >= 0).all(axis=1)
    # Itinerario de todas las vueltas monótonas (para P; no usa horas reales)
    S, E, D = S[monotono], E[monotono], D[monotono]
    E_ok = np.where(np.isnan(D), np.nan, E)
    vueltas = times_raw.loc[monotono, ["id", "date", "lap", "vehicle_id_id"]].reset_index(drop=True)
    vueltas["salida_prog_seg"] = S[:, 0]
    vueltas["hora_salida"] = S[:, 0] / 3600
    vueltas["dur_prog"] = (S[:, 17] - S[:, 0]) / 60
    vueltas["dia_semana"] = pd.to_datetime(vueltas["date"]).dt.dayofweek
    vueltas["y"] = D[:, 17]
    vueltas["clase"] = a_clase(vueltas["y"])
    assert len(vueltas) == 4559

    # GPS (idéntico a E2)
    mon = mon_raw.drop_duplicates(["vehicle_id_id", "date", "time"]).copy()
    assert len(mon_raw) - len(mon) == 6783
    mon["t_seg"] = pd.to_timedelta(mon["time"]).dt.total_seconds()
    mon = mon.sort_values(["vehicle_id_id", "date", "t_seg"]).reset_index(drop=True)
    GPS = {k: (g.t_seg.values, g.speed.values) for k, g in mon.groupby(["vehicle_id_id", "date"])}

    def agregados_gps(k):
        filas = []
        for i in range(len(vueltas)):
            key = (vueltas.vehicle_id_id.iat[i], vueltas.date.iat[i])
            val = np.flatnonzero(~np.isnan(E_ok[i, :k]))
            if key not in GPS or len(val) == 0:
                filas.append([np.nan] * 6); continue
            ss, sp = GPS[key]
            t_ini = E_ok[i, 0] if not np.isnan(E_ok[i, 0]) else S[i, 0]
            t_fin = E_ok[i, val[-1]]
            t_ult = E_ok[i, val[-2]] if len(val) >= 2 else t_ini
            filas.append(_agregados(ss, sp, t_ini, t_fin) + _agregados(ss, sp, t_ult, t_fin))
        return pd.DataFrame(filas, columns=["gps_vel_media", "gps_vel_desv", "gps_min_detenido",
                                            "gps_vel_media_ult", "gps_vel_desv_ult", "gps_min_detenido_ult"])

    GPS_K = {k: agregados_gps(k) for k in K_VALUES if k > 0}
    C = contexto_dia(vueltas, E_ok, S)
    P = planificacion(S[:, 0], vueltas["date"].values, vueltas)
    A = anomalia_dia(vueltas, E_ok, S)

    ok = vueltas.y.notna().values
    fechas = vueltas["date"].values
    vm = vueltas[ok].reset_index(drop=True)
    d = {"vueltas": vueltas, "S": S, "E_ok": E_ok, "D": D, "ok": ok, "vm": vm, "Dm": D[ok],
         "GPS_K": {k: g[ok].reset_index(drop=True) for k, g in GPS_K.items()},
         "C": C[ok].reset_index(drop=True), "P": P[ok].reset_index(drop=True), "A": A[ok].reset_index(drop=True),
         "es_test": (fechas >= TEST_INICIO)[ok], "es_dev": ((fechas >= DEV_INICIO) & (fechas < TEST_INICIO))[ok]}
    f = vm["date"].values
    d["FOLDS"] = [(np.flatnonzero((f >= DEV_INICIO) & (f <= a)), np.flatnonzero((f >= b) & (f <= c))) for a, b, c in CV_FOLDS]
    d["IDX_DEV"], d["IDX_TEST"] = np.flatnonzero(d["es_dev"]), np.flatnonzero(d["es_test"])
    assert len(vm) == 4004 and len(d["IDX_TEST"]) == 758 and len(d["IDX_DEV"]) == 3246
    assert [(len(a), len(b)) for a, b in d["FOLDS"]] == [(1503, 451), (1954, 377), (2331, 505), (2836, 410)]
    print(f"Preparación terminada en {time.time() - t0:.0f} s")
    return d


def cargar(rehacer=False):
    if os.path.exists(CACHE) and not rehacer:
        with open(CACHE, "rb") as fh:
            return pickle.load(fh)
    d = _preparar()
    with open(CACHE, "wb") as fh:
        pickle.dump(d, fh)
    return d


if __name__ == "__main__":
    cargar(rehacer=True)
