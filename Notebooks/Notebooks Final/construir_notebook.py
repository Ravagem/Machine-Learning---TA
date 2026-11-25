"""Genera y ejecuta AM2026_2_Final.ipynb (notebook de la entrega final) a partir de los módulos .py."""
import os
import nbformat as nbf
from nbclient import NotebookClient

AQUI = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook()
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

M("""
# Predicción de la impuntualidad en una ruta de transporte público de Cusco
## Notebook de la entrega final

**Curso:** 1INF02 Aprendizaje de Máquina — 2026-2 (PUCP) · **Grupo 06:** Jafeth Manases Cuyo Alata, Raul Esteban Herrera Guerrero, Matias Esteban Marrufo Zapata

Este notebook continúa el del Entregable 2 (`Notebooks E2/AM2026_2_Experimentacion.ipynb`) y documenta las mejoras de la entrega final.
El código está en módulos `.py` de esta carpeta para que cada paso se pueda leer y probar por separado:

| Módulo | Contenido |
|---|---|
| `prep.py` | Limpieza, variables F0/F1/F2 y partición temporal (idénticas a E2) + grupos nuevos C, A y P |
| `modelos.py` | Modelos, Pipeline y métricas (E2 + exactitud, exactitud balanceada, sensibilidad/especificidad) |
| `cribado.py` | Cribado de los grupos de variables nuevos (solo validación cruzada) |
| `experimento.py` | Búsqueda en malla con validación cruzada temporal (malla ampliada de XGBoost; umbrales −1/+5 y −3/+10) |
| `evaluacion.py` | Selección por validación, evaluación única en prueba, pruebas estadísticas, pesos por clase, importancia |
| `figuras_eda.py`, `figuras_resultados.py` | Gráficos de la presentación (incluye SHAP) |

Las reglas de decisión se fijaron **antes** de ver resultados nuevos en `PROTOCOLO.md`.

**Ejecución local:** colocar `Times.csv` y `Monitoring.csv` en `Dataset/` (o definir `DATA_DIR`). La búsqueda en malla tarda ~1 h con 8 núcleos; si `resultados/cv_raw.csv` existe, se reutiliza.
""")

C("""
import os, sys, json, platform
import numpy as np, pandas as pd
import sklearn, xgboost, shap, matplotlib
from IPython.display import Image, display, Markdown
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
print("python", platform.python_version(), "| pandas", pd.__version__, "| numpy", np.__version__, "| scikit-learn", sklearn.__version__,
      "| xgboost", xgboost.__version__, "| shap", shap.__version__, "| matplotlib", matplotlib.__version__)
R = lambda f: pd.read_csv(os.path.join("resultados", f))
FIG = lambda f: display(Image(os.path.join("fig", f), width=900))
""")

M("""
## 1. Datos y partición (sin cambios respecto de E2)

`prep.cargar()` reproduce la limpieza del Entregable 2 y verifica con `assert` las mismas cifras: 4 707 → 4 559 → 4 004 vueltas,
758 de prueba (28/01–01/02, **incluye el sábado 01/02**), 3 246 de desarrollo y los mismos 4 pliegues de ventana expansiva.
""")
C("""
from prep import cargar
d = cargar()
print("vueltas de modelado:", len(d["vm"]), "| desarrollo:", len(d["IDX_DEV"]), "| prueba:", len(d["IDX_TEST"]))
print("pliegues (entrenamiento/validación):", [(len(a), len(b)) for a, b in d["FOLDS"]])
""")
for f in ["pres_01_mapa_ruta.png", "pres_02_perfil_franja.png", "pres_03_calor_hora_dia.png", "pres_04_calendario_particion.png",
          "pres_05_distribucion_clases.png"]:
    C(f'FIG("{f}")')

M("""
## 2. Mejora 1 — variables del día en curso (cribado)

**Hipótesis:** el sábado concentra el 28 % del error; si el modelo supiera "cómo va el día", podría corregirse.

**Regla anti-fuga:** para predecir la vuelta *i* solo se usan vueltas del mismo día cuya **llegada real a PC18 es anterior a la hora
programada de salida** de *i*.

| Grupo | Variables |
|---|---|
| C | retraso medio de las vueltas del día ya terminadas; media de las 5 últimas; n.º de terminadas; retraso de la vuelta anterior del mismo vehículo |
| P | n.º de salidas programadas en los 60 min previos; minutos desde la salida programada anterior (itinerario, conocido de antemano) |
| A | (segundo intento declarado) anomalía del día: retraso de las terminadas **menos** el promedio histórico de su hora, calculado solo con días anteriores |

**Regla de aceptación (D1, fijada antes):** mejorar la media de validación **y** ganar en ≥ 3 de 4 pliegues. XGBoost con los
hiperparámetros elegidos en E2, en k = 0 y k = 12. La prueba no se usa.
""")
C("""
cr = R("cribado_CP_-1_5.csv")
cr[["tarea", "k", "grupo", "metrica", "media", "std", "p1", "p2", "p3", "p4", "pliegues_ganados", "dif_media", "acepta_D1"]].round(3)
""")
M("""
**Decisión:** C, P y C+P empeoran la validación en los 4 casos. A pasa la regla solo en regresión con k = 12 (−0,037 min) y empeora
en los otros 3 casos; esa mejora está en el orden del ruido de plataforma (ver sección 3), y adoptarla solo donde funcionó sería
elegir a posteriori. **Ninguna variable del día se incorpora.** Explicación: con 10 a 19 días de entrenamiento por pliegue,
el modelo ve muy pocos "estados del día" distintos para aprender de ellos; es trabajo futuro con más meses de datos.
""")

M("""
## 3. Mejora 2 — malla ampliada de XGBoost y umbrales de clase

* **Malla ampliada (D5):** en E2 XGBoost eligió profundidad 3 y tasa 0,03, en el borde de la malla. Se agregan profundidad 2,
  tasa 0,01 y 1 600 árboles (la malla original queda contenida). Random Forest y los lineales mantienen su malla.
* **Umbrales (D4):** se mantienen −1/+5 min (Cao et al.; el rango "1 min antes – 5 min después" es el más usado en la práctica).
  Como análisis de sensibilidad se repite la clasificación con −3/+10, reportando siempre la exactitud de la clase mayoritaria.
* **Reproducibilidad:** XGBoost con submuestreo < 1 varía levemente con el número de hilos y la plataforma (hasta 0,03 min por
  pliegue). Por eso toda la búsqueda se rehízo en una sola máquina, con 1 hilo por modelo (resultado determinista).
""")
C("""
cv = R("cv_raw.csv")
print("combinaciones evaluadas:", len(cv), "(cada una en 4 pliegues)")
cv.groupby(["tarea", "umbral", "modelo"]).size().unstack()
""")
C("""
mej, ori = R("cv_mejores.csv"), R("cv_mejores_malla_original.csv")
x = mej[mej.modelo == "xgb"].merge(ori[ori.modelo == "xgb"], on=["tarea", "umbral", "modelo", "k", "conjunto"], suffixes=("_ampl", "_orig"))
x[["tarea", "umbral", "k", "conjunto", "metrica_media_orig", "metrica_media_ampl", "params_ampl"]].round(4)
""")

M("## 4. Validación cruzada (selección de modelos)")
C("""
base = R("cv_lineas_base.csv")
def tabla_cv(tarea, umbral):
    t = mej[(mej.tarea == tarea) & (mej.umbral == umbral)].copy()
    t["esc"] = "k=" + t.k.astype(str) + " " + t.conjunto
    t["v"] = t.metrica_media.map("{:.3f}".format) + " ± " + t.metrica_std.map("{:.3f}".format)
    return t.pivot(index="modelo", columns="esc", values="v")[["k=0 F0", "k=4 F1", "k=8 F1", "k=12 F1", "k=4 F2", "k=8 F2", "k=12 F2"]]
display(Markdown("**MAE de validación (min)**")); display(tabla_cv("reg", "-"))
display(Markdown("**F1-macro de validación (−1/+5)**")); display(tabla_cv("clf", "cao"))
display(Markdown("**Líneas base en validación**")); display(base.round(3))
""")
C('FIG("pres_11_pliegues_validacion.png")')

M("## 5. Evaluación única en el conjunto de prueba")
C("""
res = R("test_resultados.csv")
def tabla(tarea, umbral, met):
    t = res[(res.tarea == tarea) & (res.umbral == umbral)].copy(); t["esc"] = "k=" + t.k.astype(str) + " " + t.conjunto
    return t.pivot(index="modelo", columns="esc", values=met)[["k=0 F0", "k=4 F1", "k=8 F1", "k=12 F1", "k=4 F2", "k=8 F2", "k=12 F2"]].round(3)
for met in ("MAE", "RMSE", "R2"):
    display(Markdown(f"**{met} en prueba**")); display(tabla("reg", "-", met))
for met in ("F1_macro", "exactitud", "exactitud_bal", "kappa", "recall_retrasado", "PR_AUC"):
    display(Markdown(f"**{met} en prueba (−1/+5)**")); display(tabla("clf", "cao", met))
""")
C("""
display(Markdown("**Sensibilidad y especificidad por clase (XGBoost, prueba, −1/+5)**"))
c = res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "xgb")]
c[["k", "conjunto", "exactitud", "exactitud_bal", "kappa", "sens_adel", "espec_adel", "sens_punt", "espec_punt", "sens_retr", "espec_retr"]].round(3)
""")
for f in ["pres_10_mae_vs_k.png", "pres_12_predicho_vs_real.png", "pres_13_error_por_dia.png", "pres_14_confusion.png", "pres_15_sens_espec.png"]:
    C(f'FIG("{f}")')

M("## 6. Pruebas de hipótesis (Wilcoxon / McNemar, α = 0,05)")
C("""
for f, t in (("test_h1.csv", "H1: mejor modelo vs líneas base"), ("test_h2.csv", "H2: efecto de k"), ("test_h3.csv", "H3: F1 vs F2 (GPS)"),
             ("test_xgb_vs_rf.csv", "D6: XGBoost vs Random Forest (Holm por 7 escenarios)")):
    display(Markdown(f"**{t}**")); display(R(f).round(4))
""")
C('FIG("pres_17_xgb_vs_rf.png")')

M("""
## 7. Mejora 3 — pesos por clase para recuperar "adelantado" (D4b)

Se multiplican las probabilidades de "adelantado" y "retrasado" por pesos (0,5 a 3,0) y se predice la clase de mayor probabilidad
ponderada. **Validación anidada:** los pesos se eligen con 3 pliegues y se evalúan en el cuarto, rotando; la regla D1 decide.
""")
C("""
display(R("d4b_pesos_validacion.csv").round(3))
display(Markdown("**En prueba (solo se adopta si pasó D1 en validación)**")); display(R("d4b_pesos_test.csv").round(3))
""")

M("## 8. Análisis de sensibilidad de los umbrales de clase")
C("""
s = res[(res.tarea == "clf") & (res.modelo.isin(["xgb", "B2 histórico"])) & (res.k == 0)]
s[["umbral", "modelo", "exactitud", "exactitud_bal", "F1_macro", "kappa", "sens_adel", "sens_retr"]].round(3)
""")
C('FIG("pres_16_sensibilidad_umbrales.png")')

M("## 9. Interpretación: importancia por permutación y SHAP")
C("""
imp = R("importancia_permutacion.csv")
display(imp.sort_values("media", ascending=False).groupby(["tarea", "k", "conjunto"]).head(6).round(4))
""")
for f in ["pres_18_importancia.png", "pres_19_shap_resumen.png", "pres_20_shap_sabado.png", "pres_20_shap_madrugada.png"]:
    C(f'FIG("{f}")')

M("""
## 10. Archivos generados

* `resultados/`: tablas en CSV (validación, prueba, pruebas estadísticas, predicciones por vuelta).
* `modelos/`: modelos XGBoost entrenados con el desarrollo (`joblib`), para la demo.
* `fig/`: gráficos de la presentación (`pres_*.png`).
""")

ruta = os.path.join(AQUI, "AM2026_2_Final.ipynb")
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": AQUI}}).execute()
nbf.write(nb, ruta)
print("Notebook escrito y ejecutado:", ruta)
