"""Gráficos de resultados para la presentación (16:9). Requiere haber corrido experimento.py y evaluacion.py."""
import json, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figuras_eda import (guardar, FIG, AZUL, NARANJA, AQUA, ROJO, TINTA, TINTA2, MUTED, BASE, SUP, W, H, SECUENCIAL)
from prep import cargar
from modelos import construir_features

AQUI = os.path.dirname(os.path.abspath(__file__))
R = lambda f: pd.read_csv(os.path.join(AQUI, "resultados", f))
COL = {"xgb": AZUL, "rf": NARANJA, "ridge": AQUA, "logistic": AQUA}
NOM = {"xgb": "XGBoost", "rf": "Random Forest", "ridge": "Ridge", "logistic": "Logística",
       "B1 persistencia": "B1 persistencia", "B2 histórico": "B2 promedio histórico"}
coma = lambda v, n=2: f"{v:.{n}f}".replace(".", ",")


def main():
    res = R("test_resultados.csv")
    pred = R("test_predicciones.csv")
    d = cargar()
    vm = d["vm"]

    # 10. MAE en prueba frente a k
    t = res[(res.tarea == "reg") & (res.conjunto != "F2")]
    fig, ax = plt.subplots(figsize=(W, H))
    for mo in ["B2 histórico", "B1 persistencia", "ridge", "rf", "xgb"]:
        g = t[t.modelo == mo].sort_values("k")
        base = mo.startswith("B")
        ax.plot(g.k, g.MAE, ls="--" if base else "-", color=MUTED if base else COL[mo], lw=2 if base else 3,
                marker="o", ms=9, mec=SUP, mew=2)
        ax.annotate(NOM[mo], (g.k.iat[-1], g.MAE.iat[-1]), xytext=(14, 0), textcoords="offset points", va="center",
                    fontsize=13, color=TINTA)
    gx = t[t.modelo == "xgb"].sort_values("k")
    for k, v in zip(gx.k, gx.MAE):
        ax.annotate(coma(v), (k, v), xytext=(-30, -22) if k == 12 else (0, -24), textcoords="offset points", ha="center",
                    fontsize=13, color=AZUL, weight="bold")
    ax.set_xticks([0, 4, 8, 12]); ax.set_xticklabels(["0\n(antes de salir)", "4", "8", "12"]); ax.set_xlim(-0.8, 15.5)
    ax.set_xlabel("Puntos de control ya observados (k)"); ax.set_ylabel("MAE en prueba (min)  ↓ mejor")
    b2 = t[(t.modelo == "B2 histórico") & (t.k == 0)].MAE.iat[0]; x0 = gx.MAE.iat[0]
    ax.set_title(f"Antes de salir, XGBoost ya tiene {100 * (1 - x0 / b2):.0f} % menos error que el promedio histórico")
    guardar(fig, "pres_10_mae_vs_k.png", "Conjunto de prueba (758 vueltas, 28/01–01/02). F0 en k = 0 y F1 en k > 0.")

    # 11. Variabilidad entre pliegues (k = 0, regresión) y valor en prueba
    mej = R("cv_mejores.csv"); base = R("cv_lineas_base.csv")
    fig, ax = plt.subplots(figsize=(W, H * 0.9))
    filas = [("B2 histórico", None), ("ridge", "ridge"), ("rf", "rf"), ("xgb", "xgb")]
    for i, (nom, mo) in enumerate(filas):
        if mo is None:
            continue
        r = mej[(mej.tarea == "reg") & (mej.modelo == mo) & (mej.k == 0)].iloc[0]
        v = r[["pliegue1", "pliegue2", "pliegue3", "pliegue4"]].values.astype(float)
        ax.scatter(v, [i] * 4, s=110, color=COL[mo], alpha=0.55, edgecolors=SUP, linewidths=1.5, zorder=3)
        ax.plot([v.mean()] * 2, [i - 0.28, i + 0.28], color=COL[mo], lw=3)
        te = res[(res.tarea == "reg") & (res.modelo == mo) & (res.k == 0)].MAE.iat[0]
        ax.scatter([te], [i], marker="D", s=120, color=TINTA, zorder=4)
        ax.annotate(f"validación {coma(v.mean())} ± {coma(v.std(ddof=1))}   ·   prueba {coma(te)}", (v.max(), i),
                    xytext=(16, 0), textcoords="offset points", va="center", fontsize=12, color=TINTA2)
    rb = base[(base.tarea == "reg") & (base.modelo == "B2 histórico") & (base.k == 0)]
    teb = res[(res.tarea == "reg") & (res.modelo == "B2 histórico") & (res.k == 0)].MAE.iat[0]
    ax.plot([rb["mean"].iat[0]] * 2, [-0.28, 0.28], color=MUTED, lw=3); ax.scatter([teb], [0], marker="D", s=120, color=TINTA, zorder=4)
    ax.annotate(f"validación {coma(rb['mean'].iat[0])} ± {coma(rb['std'].iat[0])}   ·   prueba {coma(teb)}", (rb["mean"].iat[0], 0),
                xytext=(16, 10), textcoords="offset points", va="center", fontsize=12, color=TINTA2)
    ax.set_yticks(range(4)); ax.set_yticklabels([NOM[f[0]] for f in filas]); ax.set_xlim(1.8, 8.8)
    ax.set_xlabel("MAE (min)  ← mejor"); ax.grid(axis="y", visible=False)
    ax.set_title("La dispersión entre pliegues es grande: por eso exigimos 3 de 4")
    guardar(fig, "pres_11_pliegues_validacion.png", "k = 0. Círculos: los 4 pliegues; barra: media; rombo negro: prueba. El pliegue con el sábado 25/01 es el más difícil.")

    # 12. Predicho vs real (XGBoost, k = 0) — sábado destacado
    p = pred[(pred.tarea == "reg") & (pred.modelo == "xgb") & (pred.k == 0)].copy()
    p["sab"] = pd.to_datetime(p.date).dt.dayofweek == 5
    fig, ax = plt.subplots(figsize=(W * 0.62, H))
    lim = (-30, 40)
    ax.fill_between(lim, [lim[0] - 5, lim[1] - 5], [lim[0] + 5, lim[1] + 5], color=BASE, alpha=0.35, lw=0)
    ax.plot(lim, lim, color=TINTA2, lw=1)
    for sab, c, lab in ((False, AZUL, "Martes a viernes"), (True, NARANJA, "Sábado 01/02")):
        q = p[p.sab == sab]
        ax.scatter(q.pred, q.real, s=26, color=c, alpha=0.6, edgecolors=SUP, linewidths=0.6,
                   label=f"{lab}: MAE {coma((q.real - q.pred).abs().mean())} min ({len(q)} vueltas)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("Retraso predicho antes de salir (min)"); ax.set_ylabel("Retraso real (min)")
    ax.legend(loc="upper left", fontsize=12)
    ax.set_title("Predicción antes de salir frente a lo ocurrido")
    guardar(fig, "pres_12_predicho_vs_real.png", "XGBoost, k = 0, prueba. Franja gris: error de ±5 min.")

    # 13. Error por día de prueba
    e = R("test_error_por_dia.csv")
    e["dow"] = pd.to_datetime(e.date).dt.dayofweek
    DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, ax = plt.subplots(figsize=(W, H * 0.9))
    ax.bar(range(len(e)), e.MAE, color=[NARANJA if w == 5 else AZUL for w in e.dow], width=0.6)
    for i, r in e.iterrows():
        ax.text(i, r.MAE + 0.12, f"MAE {coma(r.MAE)}\n{r.vueltas} vueltas · {coma(r.pct_error, 0)} % del error", ha="center",
                fontsize=12, color=TINTA)
    ax.set_xticks(range(len(e))); ax.set_xticklabels([f"{DIAS[w]} {f[8:]}/{f[5:7]}" for f, w in zip(e.date, e.dow)])
    ax.set_ylim(0, e.MAE.max() * 1.3); ax.set_ylabel("MAE (min)")
    ax.set_title("El sábado concentra el error: el modelo subestima su retraso extra")
    guardar(fig, "pres_13_error_por_dia.png", "XGBoost, k = 0, conjunto de prueba.")

    # 14. Matrices de confusión normalizadas por fila (recall)
    ETQ = ["Adelantado", "Puntual", "Retrasado"]
    fig, axs = plt.subplots(1, 2, figsize=(W, H))
    for ax, k, c in zip(axs, (0, 12), ("F0", "F1")):
        q = pred[(pred.tarea == "clf") & (pred.umbral == "cao") & (pred.modelo == "xgb") & (pred.k == k) & (pred.conjunto == c)]
        cm = pd.crosstab(q.real, q.pred).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values
        rn = cm / cm.sum(1, keepdims=True)
        ax.imshow(rn, cmap=SECUENCIAL, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * rn[i, j]:.0f} %\n({cm[i, j]})", ha="center", va="center", fontsize=14,
                        color="white" if rn[i, j] > 0.55 else TINTA)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(ETQ); ax.set_yticklabels(ETQ)
        ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real"); ax.grid(False)
        [s.set_visible(False) for s in ax.spines.values()]
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"k = {k} · exactitud {coma(100 * acc, 1)} %", fontsize=16)
    fig.suptitle("Los errores son entre clases vecinas; casi nunca adelanto ↔ retraso",
                 x=0.012, ha="left", fontsize=19, fontweight="bold")
    guardar(fig, "pres_14_confusion.png", "XGBoost, prueba. Cada fila suma 100 % (sensibilidad de cada clase en la diagonal); entre paréntesis, n.º de vueltas.")

    # 15. Sensibilidad y especificidad por clase, y exactitud frente a la clase mayoritaria
    r0 = res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "xgb") & (res.k == 0)].iloc[0]
    rb = res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "B2 histórico") & (res.k == 0)].iloc[0]
    fig, axs = plt.subplots(1, 2, figsize=(W, H), gridspec_kw={"width_ratios": [2.1, 1]})
    ax = axs[0]
    x = np.arange(3); wb = 0.36
    sens = [r0.sens_adel, r0.sens_punt, r0.sens_retr]; esp = [r0.espec_adel, r0.espec_punt, r0.espec_retr]
    b1 = ax.bar(x - wb / 2 - 0.01, sens, wb, color=AZUL, label="Sensibilidad (recall)")
    b2 = ax.bar(x + wb / 2 + 0.01, esp, wb, color=NARANJA, label="Especificidad")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, coma(b.get_height()), ha="center", fontsize=13, color=TINTA)
    ax.set_xticks(x); ax.set_xticklabels(ETQ); ax.set_ylim(0, 1.12); ax.legend(loc="upper center", ncol=2)
    ax.set_title("Por clase (uno contra el resto)", fontsize=15, fontweight="normal", color=TINTA2)
    ax = axs[1]
    vals = [("Exactitud", r0.exactitud, rb.exactitud), ("Exactitud\nbalanceada", r0.exactitud_bal, rb.exactitud_bal), ("Kappa", r0.kappa, rb.kappa)]
    for i, (nom, v, vb) in enumerate(vals):
        ax.barh(i, v, color=AZUL, height=0.5)
        ax.plot([vb, vb], [i - 0.32, i + 0.32], color=TINTA, lw=2.5)
        ax.text(v + 0.02, i, coma(v), va="center", fontsize=13, color=TINTA)
        ax.text(vb, i + 0.38, f"mayoritaria {coma(vb)}", ha="center", fontsize=10.5, color=TINTA2)
    ax.set_yticks(range(3)); ax.set_yticklabels([v[0] for v in vals]); ax.set_xlim(0, 1.05); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("Global vs. clase mayoritaria", fontsize=15, fontweight="normal", color=TINTA2)
    fig.suptitle("Alta especificidad en las tres clases; el adelanto es el más difícil",
                 x=0.012, ha="left", fontsize=18, fontweight="bold")
    guardar(fig, "pres_15_sens_espec.png", "XGBoost, k = 0, prueba, umbrales −1/+5 min. Línea negra: clase mayoritaria (predecir siempre 'puntual').")

    # 16. Análisis de sensibilidad de umbrales
    fig, ax = plt.subplots(figsize=(W, H * 0.9))
    mets = [("exactitud", "Exactitud"), ("exactitud_bal", "Exactitud balanceada"), ("F1_macro", "F1-macro"), ("kappa", "Kappa")]
    x = np.arange(len(mets))
    for j, (u, c, lab) in enumerate((("cao", AZUL, "−1 / +5 (Cao et al., principal)"), ("amplio", NARANJA, "−3 / +10 (tolerancia amplia)"))):
        rm = res[(res.tarea == "clf") & (res.umbral == u) & (res.modelo == "xgb") & (res.k == 0)].iloc[0]
        rmay = res[(res.tarea == "clf") & (res.umbral == u) & (res.modelo == "B2 histórico") & (res.k == 0)].iloc[0]
        xs = x + (j - 0.5) * 0.38
        ax.bar(xs, [rm[m] for m, _ in mets], 0.36, color=c, label=lab)
        for xi, (m, _) in zip(xs, mets):
            ax.text(xi, rm[m] + 0.015, coma(rm[m]), ha="center", fontsize=12.5, color=TINTA)
            ax.plot([xi - 0.17, xi + 0.17], [rmay[m]] * 2, color=TINTA, lw=2.5)
    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in mets]); ax.set_ylim(0, 1.08); ax.legend(loc="upper right")
    ax.set_title("Umbrales tolerantes inflan la exactitud; el kappa no cambia")
    guardar(fig, "pres_16_sensibilidad_umbrales.png", "XGBoost, k = 0, prueba. Línea negra: clase mayoritaria con esos umbrales.")

    # 17. XGBoost vs Random Forest
    xr = R("test_xgb_vs_rf.csv")
    q = xr[xr.tarea == "reg"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(W, H * 0.9))
    for i, r in q.iterrows():
        ax.plot([r.XGB, r.RF], [i, i], color=BASE, lw=3, zorder=1)
        ax.scatter([r.XGB], [i], s=130, color=AZUL, zorder=3, edgecolors=SUP, linewidths=1.5)
        ax.scatter([r.RF], [i], s=130, color=NARANJA, zorder=3, edgecolors=SUP, linewidths=1.5)
        ax.text(max(r.XGB, r.RF) + 0.03, i, f"p (Holm) = {coma(r.p_holm, 3)}" + ("  *" if r.significativo_holm else ""),
                va="center", fontsize=12, color=TINTA2)
    ax.set_yticks(range(len(q))); ax.set_yticklabels([f"k = {r.k}, {r.conjunto}" for r in q.itertuples()]); ax.invert_yaxis()
    ax.scatter([], [], color=AZUL, s=100, label="XGBoost"); ax.scatter([], [], color=NARANJA, s=100, label="Random Forest")
    ax.legend(loc="lower right"); ax.set_xlabel("MAE en prueba (min)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(q[["XGB", "RF"]].values.min() - 0.1, q[["XGB", "RF"]].values.max() + 0.45)
    ax.set_title("XGBoost solo supera a Random Forest con 12 PC observados")
    guardar(fig, "pres_17_xgb_vs_rf.png", "Wilcoxon sobre errores absolutos pareados, corrección de Holm por 7 escenarios. * = significativo (α = 0,05).")

    # 18. Importancia por permutación (regresión, k = 0)
    imp = R("importancia_permutacion.csv")
    q = imp[(imp.tarea == "reg") & (imp.k == 0)].sort_values("media").tail(8)
    NV = {"hora_salida": "Hora programada de salida", "dia_semana": "Día de la semana", "dur_prog": "Duración programada (horario)",
          "vehiculo": "Vehículo", "lap": "N.º de vuelta del día"}
    fig, ax = plt.subplots(figsize=(W, H * 0.8))
    ax.barh([NV.get(v, v) for v in q.variable], q.media, xerr=q["std"], color=AZUL, height=0.55, error_kw={"lw": 1.2, "ecolor": TINTA2})
    for i, v in enumerate(q.media):
        ax.text(v + 0.08, i, f"+{coma(v)} min", va="center", fontsize=13, color=TINTA)
    ax.set_xlabel("Aumento del MAE al desordenar la variable (min)"); ax.grid(axis="y", visible=False)
    ax.set_title("La hora y el día explican casi todo: el retraso es sistemático")
    guardar(fig, "pres_18_importancia.png", "Importancia por permutación, XGBoost, k = 0, prueba, 10 repeticiones (barra: desviación estándar).")

    # 19. SHAP: resumen global y explicación de dos vueltas
    import shap
    art = joblib.load(os.path.join(AQUI, "modelos", "xgb_reg_k0_F0.joblib"))
    X = construir_features(vm, d["Dm"], 0, "F0")[art["columnas"]]
    Xt = X.iloc[d["IDX_TEST"]]
    expl = shap.TreeExplainer(art["modelo"])
    sv = expl(Xt)
    sv.feature_names = [NV.get(c, c) for c in Xt.columns]
    fig = plt.figure(figsize=(W, H))
    shap.plots.beeswarm(sv, show=False, max_display=5, color_bar_label="Valor de la variable")
    plt.gcf().set_size_inches(W, H)
    plt.title("SHAP: qué empuja la predicción de cada vuelta", loc="left", fontsize=18, fontweight="bold")
    plt.xlabel("Contribución al retraso predicho (min)")
    guardar(plt.gcf(), "pres_19_shap_resumen.png", "XGBoost, k = 0, 758 vueltas de prueba. Cada punto es una vuelta; color: valor de la variable (rojo alto, azul bajo).")
    vt = vm.iloc[d["IDX_TEST"]].reset_index(drop=True)
    casos = {"sabado": int(((vt.dia_semana == 5) & (vt.hora_salida.between(16, 18))).idxmax()),
             "madrugada": int(((vt.dia_semana == 2) & (vt.hora_salida < 5)).idxmax())}
    for nom, i in casos.items():
        fig = plt.figure(figsize=(W, H))
        shap.plots.waterfall(sv[i], show=False, max_display=6)
        plt.gcf().set_size_inches(W, H)
        r = vt.iloc[i]
        h = f"{int(r.hora_salida)}:{int(round(r.hora_salida % 1 * 60)):02d}"
        plt.title(f"Vuelta del {r.date[8:]}/{r.date[5:7]} a las {h}: predicho {coma(sv[i].values.sum() + sv[i].base_values, 1)} min, "
                  f"real {coma(r.y, 0)} min", loc="left", fontsize=17, fontweight="bold")
        guardar(plt.gcf(), f"pres_20_shap_{nom}.png", "Parte del promedio del entrenamiento E[f(X)] y suma la contribución de cada variable "
                "(rojo: aumenta el retraso; azul: lo reduce). Día: 0 = lunes … 5 = sábado.")


if __name__ == "__main__":
    main()
