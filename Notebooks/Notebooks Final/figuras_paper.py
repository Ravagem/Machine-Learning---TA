"""Figuras del paper final a ancho de una columna IEEE (3,4 pulgadas), mismo estilo que las del Entregable 2.
Las figuras 1–4 (exploratorias) no cambian y se copian de Notebooks E2/fig. Salida: Informes/Paper_Final/fig/"""
import os, shutil
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prep import cargar
from modelos import construir_features

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "..", "Informes", "Paper_Final", "fig")
os.makedirs(OUT, exist_ok=True)
R = lambda f: pd.read_csv(os.path.join(AQUI, "resultados", f))
AZUL, NARANJA, AQUA, ROJO, GRIS, TINTA2 = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#898781", "#52514e"
plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": TINTA2, "xtick.color": TINTA2, "ytick.color": TINTA2, "savefig.dpi": 300,
                     "legend.frameon": False})


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, nombre), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("  ", nombre)


for f in ("paper_fig1_ruta.png", "paper_fig2_hora_dia.png", "paper_fig3_tramos.png", "paper_fig4_corr_k.png"):
    shutil.copy(os.path.join(AQUI, "..", "Notebooks E2", "fig", f), os.path.join(OUT, f))

res = R("test_resultados.csv")
COL = {"xgb": (AZUL, "^", "XGBoost"), "rf": (NARANJA, "s", "Random Forest"), "ridge": (AQUA, "o", "Ridge"),
       "logistic": (AQUA, "o", "Logística"), "B1 persistencia": (ROJO, "x", "B1 persistencia"),
       "B2 histórico": (GRIS, "+", "B2 histórico/mayoritaria")}

# Fig. 5 — efecto de k en prueba (MAE y F1-macro)
fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.6))
for ax, t, u, met, lab, tit in ((axs[0], "reg", "-", "MAE", "MAE en prueba (min)", "(a) Regresión"),
                                (axs[1], "clf", "cao", "F1_macro", "F1-macro en prueba", "(b) Clasificación")):
    q = res[(res.tarea == t) & (res.umbral == u) & (res.conjunto != "F2")]
    for mo, g in q.groupby("modelo"):
        g = g.sort_values("k"); c, mk, nom = COL[mo]
        ax.plot(g.k, g[met], "--" if mo.startswith("B") else "-", marker=mk, color=c, ms=3.5, lw=1, label=nom)
    ax.set_xticks([0, 4, 8, 12]); ax.set_ylabel(lab); ax.set_title(tit, fontsize=7, loc="left")
axs[1].set_xlabel("Puntos de control observados (k)")
h1, l1 = axs[0].get_legend_handles_labels(); h2, l2 = axs[1].get_legend_handles_labels()
ley = dict(zip(l1 + l2, h1 + h2))
fig.legend(ley.values(), ley.keys(), fontsize=5.8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join(OUT, "paper_fig5_efecto_k.png"), dpi=300, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("   paper_fig5_efecto_k.png")

# Fig. 6 — matrices de confusión de XGBoost (k = 0 y k = 12 con pesos por clase)
pred = R("test_predicciones.csv")
fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.75))
etq = ["Adel.", "Punt.", "Retr."]
for ax, (mo, k, tit) in zip(axs, (("xgb", 0, "k = 0"), ("xgb+pesos", 12, "k = 12 (F1, con pesos)"))):
    q = pred[(pred.tarea == "clf") & (pred.umbral == "cao") & (pred.modelo == mo) & (pred.k == k) & (pred.conjunto.isin(["F0", "F1"]))]
    cm = pd.crosstab(q.real, q.pred).reindex(index=[0, 1, 2], columns=[0, 1, 2], fill_value=0).values
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5, color="white" if cm[i, j] > 200 else "black")
    ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(etq, fontsize=6); ax.set_yticklabels(etq, fontsize=6)
    ax.set_xlabel("Predicha", fontsize=6.5); ax.set_ylabel("Real", fontsize=6.5); ax.set_title(tit, fontsize=7)
    [s.set_visible(False) for s in ax.spines.values()]
guardar(fig, "paper_fig6_confusion.png")

# Fig. 7 — importancia por permutación
imp = R("importancia_permutacion.csv")
NV = {"hora_salida": "hora de salida", "dia_semana": "día de la semana", "dur_prog": "horario (duración)",
      "vehiculo": "vehículo", "lap": "n.º de vuelta"}
tit = {("reg", 0, "F0"): "(a) Regresión, k = 0 (pérdida de MAE, min)", ("reg", 12, "F2"): "(b) Regresión, k = 12, F2 (pérdida de MAE, min)",
       ("clf", 12, "F2"): "(c) Clasificación, k = 12, F2 (pérdida de F1-macro)"}
fig, axs = plt.subplots(3, 1, figsize=(3.4, 4.6), gridspec_kw={"height_ratios": [5, 8, 8]})
for ax, (caso, t) in zip(axs, tit.items()):
    q = imp[(imp.tarea == caso[0]) & (imp.k == caso[1]) & (imp.conjunto == caso[2])].sort_values("media", ascending=False).head(8)[::-1]
    ax.barh([NV.get(v, v) for v in q.variable], q.media, xerr=q["std"], color=AZUL, height=0.6, error_kw={"lw": 0.6})
    ax.set_title(t, fontsize=7, loc="left"); ax.tick_params(labelsize=6)
guardar(fig, "paper_fig7_importancia.png")

# Fig. 8 — SHAP (resumen global, k = 0)
import shap
d = cargar()
art = joblib.load(os.path.join(AQUI, "modelos", "xgb_reg_k0_F0.joblib"))
Xt = construir_features(d["vm"], d["Dm"], 0, "F0")[art["columnas"]].iloc[d["IDX_TEST"]]
sv = shap.TreeExplainer(art["modelo"])(Xt)
sv.feature_names = [NV.get(c, c) for c in Xt.columns]
plt.figure()
shap.plots.beeswarm(sv, show=False, max_display=5, color_bar_label="valor de la variable", plot_size=None)
f = plt.gcf(); f.set_size_inches(3.4, 1.9)
for a in f.axes:
    a.tick_params(labelsize=6); a.xaxis.label.set_size(6.5); a.yaxis.label.set_size(6)
f.axes[0].set_xlabel("Contribución SHAP al retraso predicho (min)")
guardar(f, "paper_fig8_shap.png")

# Fig. 9 — explicación de una vuelta del sábado (cascada SHAP)
vt = d["vm"].iloc[d["IDX_TEST"]].reset_index(drop=True)
i = int(((vt.dia_semana == 5) & (vt.hora_salida.between(16, 18))).idxmax())
plt.figure()
shap.plots.waterfall(sv[i], show=False, max_display=5)
f = plt.gcf(); f.set_size_inches(3.4, 1.9)
for a in f.axes:
    a.tick_params(labelsize=6)
    for t in a.texts:
        t.set_fontsize(6)
guardar(f, "paper_fig9_shap_sabado.png")
print("Vuelta del sábado explicada:", vt.iloc[i][["id", "date", "hora_salida", "y"]].to_dict(),
      "pred", float(sv[i].values.sum() + sv[i].base_values), "base", float(sv[i].base_values),
      dict(zip(Xt.columns, np.round(sv[i].values, 2))))
