"""Genera las tablas de resultados del paper final (LaTeX) directamente desde resultados/*.csv.
Salida: Informes/Paper_Final/tablas/*.tex (solo el entorno tabular; el título va en main.tex)."""
import os
import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(AQUI, "..", "..", "Informes", "Paper_Final", "tablas")
os.makedirs(OUT, exist_ok=True)
R = lambda f: pd.read_csv(os.path.join(AQUI, "resultados", f))
ESC = [(0, "F0"), (4, "F1"), (8, "F1"), (12, "F1"), (4, "F2"), (8, "F2"), (12, "F2")]
NE = lambda k, c: f"$k={k}$ {c}"


def n(v, d=2):
    if pd.isna(v):
        return "—"
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def pv(p):
    if p < 0.001:
        return "$<$ 0,001"
    if 0.045 <= p < 0.055:
        return n(p, 4)
    return n(p, 3) if p < 0.1 else n(p, 2)


def bold_best(vals, menor):
    """Devuelve índices del mejor valor (MAE: menor; F1: mayor) ignorando NaN."""
    arr = np.array([np.nan if v is None else v for v in vals], dtype=float)
    if np.all(np.isnan(arr)):
        return set()
    b = np.nanmin(arr) if menor else np.nanmax(arr)
    return {i for i, v in enumerate(arr) if not np.isnan(v) and abs(v - b) < 1e-12}


def escribir(nombre, cuerpo):
    with open(os.path.join(OUT, nombre), "w", encoding="utf8") as fh:
        fh.write(cuerpo)
    print("  ", nombre)


mej, base, res = R("cv_mejores.csv"), R("cv_lineas_base.csv"), R("test_resultados.csv")


def tabla_cv(tarea, umbral, d, menor):
    filas = []
    for k, c in ESC:
        vals, txt = [], []
        for mo in (["B1 persistencia", "B2 histórico"] + (["ridge", "rf", "xgb"] if tarea == "reg" else ["logistic", "rf", "xgb"])):
            if mo.startswith("B"):
                r = base[(base.tarea == tarea) & (base.umbral == umbral) & (base.modelo == mo) & (base.k == k)]
                if len(r) == 0:
                    vals.append(None); txt.append("—"); continue
                m, s = r["mean"].iat[0], r["std"].iat[0]
                vals.append(None)  # las líneas base no compiten por la negrita
            else:
                r = mej[(mej.tarea == tarea) & (mej.umbral == umbral) & (mej.modelo == mo) & (mej.k == k) & (mej.conjunto == c)].iloc[0]
                m, s = r.metrica_media, r.metrica_std
                vals.append(m)
            txt.append(f"{n(m, d)}$\\pm${n(s, d)}")  # sin espacios alrededor de ±: la tabla cabe en una columna a 7 pt
        b = bold_best(vals, menor)
        txt = [f"\\textbf{{{t}}}" if i in b else t for i, t in enumerate(txt)]
        filas.append(NE(k, c) + " & " + " & ".join(txt) + r" \\")
    return filas


cab = r"Escenario & B1 & B2 & Ridge & RF & XGB \\"
escribir("t_cv_mae.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n" + cab + "\n\\midrule\n" + "\n".join(tabla_cv("reg", "-", 2, True)) + "\n\\bottomrule\n\\end{tabular}\n")
cab = r"Escenario & B1 & B2 & Logística & RF & XGB \\"
escribir("t_cv_f1.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n" + cab + "\n\\midrule\n" + "\n".join(tabla_cv("clf", "cao", 3, False)) + "\n\\bottomrule\n\\end{tabular}\n")

# Prueba: MAE (+R² de XGB) y F1-macro / recall de retrasado
RT = res.set_index(["tarea", "umbral", "modelo", "k", "conjunto"])
filas = []
for k, c in ESC:
    mods = ["B1 persistencia", "B2 histórico", "ridge", "rf", "xgb"]
    vals = [RT.loc[("reg", "-", m, k, c), "MAE"] if ("reg", "-", m, k, c) in RT.index else np.nan for m in mods]
    b = bold_best([None, None] + vals[2:], True)
    txt = [n(v, 3) for v in vals]
    txt[-1] += f" ({n(RT.loc[('reg', '-', 'xgb', k, c), 'R2'])})"
    txt = [f"\\textbf{{{t}}}" if i in b else t for i, t in enumerate(txt)]
    filas.append(NE(k, c) + " & " + " & ".join(txt) + r" \\")
escribir("t_test_mae.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nEscenario & B1 & B2 & Ridge & RF & XGB ($R^2$) \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

filas = []
for k, c in ESC:
    mods = ["B1 persistencia", "B2 histórico", "logistic", "rf", "xgb"]
    f1 = [RT.loc[("clf", "cao", m, k, c), "F1_macro"] if ("clf", "cao", m, k, c) in RT.index else np.nan for m in mods]
    rc = [RT.loc[("clf", "cao", m, k, c), "recall_retrasado"] if ("clf", "cao", m, k, c) in RT.index else np.nan for m in mods]
    b = bold_best([None, None] + f1[2:], False)
    txt = ["—" if np.isnan(a) else f"{n(a)} / {n(r_)}" for a, r_ in zip(f1, rc)]
    txt = [f"\\textbf{{{t}}}" if i in b else t for i, t in enumerate(txt)]
    filas.append(NE(k, c) + " & " + " & ".join(txt) + r" \\")
escribir("t_test_f1.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nEscenario & B1 & B2 & Logística & RF & XGB \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# H1
h1 = R("test_h1.csv")
filas = []
for k, c in ESC:
    rr = h1[(h1.tarea == "reg") & (h1.k == k) & (h1.conjunto == c)]
    rc = h1[(h1.tarea == "clf") & (h1.umbral == "cao") & (h1.k == k) & (h1.conjunto == c)]
    g = lambda d_, b: d_[d_.base == b].valor_base.iat[0] if (d_.base == b).any() else np.nan
    mc = rc.valor_mejor.iat[0]; dag = "$^\\dagger$" if rc.mejor.iat[0] != "xgb" else ""
    filas.append(NE(k, c) + f" & {n(rr.valor_mejor.iat[0])} & {n(g(rr, 'B1 persistencia'))} & {n(g(rr, 'B2 histórico'))} & "
                 f"{n(mc, 3)}{dag} & {n(g(rc, 'B1 persistencia'), 3)} & {n(g(rc, 'B2 histórico'), 3)} \\\\")
pmax = h1[h1.umbral.isin(["-", "cao"])].p.max()
escribir("t_h1.tex", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{MAE (min)} & \\multicolumn{3}{c}{F1-macro} \\\\\n"
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\nEscenario & Modelo & B1 & B2 & Modelo & B1 & B2 \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")
print("   p máximo H1 (−1/+5):", pmax)

# H2
h2 = R("test_h2.csv")
filas = []
for (a, b) in ((0, 4), (4, 8), (8, 12), (0, 12)):
    v = []
    for t, mo in (("reg", "ridge"), ("reg", "rf"), ("reg", "xgb"), ("clf", "xgb")):
        p = h2[(h2.tarea == t) & (h2.modelo == mo) & (h2.de == a) & (h2.a == b)].p.iat[0]
        v.append(f"\\textbf{{{pv(p)}}}" if p < 0.05 else pv(p))
    filas.append(f"$k={a} \\rightarrow {b}$ & " + " & ".join(v) + r" \\")
escribir("t_h2.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nCambio & Ridge & RF & XGB & XGB (clasif.) \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# H3
h3 = R("test_h3.csv")
filas = []
for mo, nom in (("ridge", "Ridge/Log."), ("rf", "RF"), ("xgb", "XGB")):
    for k in (4, 8, 12):
        r = h3[(h3.tarea == "reg") & (h3.modelo == mo) & (h3.k == k)].iloc[0]
        c = h3[(h3.tarea == "clf") & (h3.modelo == ("logistic" if mo == "ridge" else mo)) & (h3.k == k)].iloc[0]
        filas.append(f"{nom} & {k} & {n(r.valor_F1)} $\\rightarrow$ {n(r.valor_F2)} & {pv(r.p)} & {n(c.valor_F1, 3)} $\\rightarrow$ {n(c.valor_F2, 3)} & {pv(c.p)} \\\\")
escribir("t_h3.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nModelo & $k$ & MAE F1 $\\rightarrow$ F2 & $p$ & F1-m F1 $\\rightarrow$ F2 & $p$ \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# Error por día
e = R("test_error_por_dia.csv")
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
filas = [f"{DIAS[pd.Timestamp(r.date).dayofweek]} {r.date[8:]}/{r.date[5:7]} & {r.vueltas} & {('+' if r.retraso_medio > 0.05 else '')}{n(r.retraso_medio, 1) if abs(r.retraso_medio) >= 0.05 else '0,0'} & {n(r.MAE)} & {n(r.pct_error, 0)} \\\\"
         for r in e.itertuples()]
escribir("t_dia.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nDía & Vueltas & Retraso medio (min) & MAE (min) & \\% del error \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# Clasificación detallada: sensibilidad / especificidad por clase
w = R("d4b_pesos_test.csv")
casos = [("XGB, $k=0$", res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "xgb") & (res.k == 0)].iloc[0]),
         ("XGB, $k=12$ F1", res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "xgb") & (res.k == 12) & (res.conjunto == "F1")].iloc[0]),
         ("XGB, $k=12$ F1 + pesos", w[(w.k == 12) & (w.conjunto == "F1") & (w.modelo == "xgb+pesos")].iloc[0]),
         ("Clase mayoritaria", res[(res.tarea == "clf") & (res.umbral == "cao") & (res.modelo == "B2 histórico") & (res.k == 0)].iloc[0])]
filas = [f"{nom} & {n(r.exactitud, 3)} & {n(r.exactitud_bal, 3)} & {n(r.kappa, 3)} & {n(r.sens_adel)} / {n(r.espec_adel)} & "
         f"{n(r.sens_punt)} / {n(r.espec_punt)} & {n(r.sens_retr)} / {n(r.espec_retr)} \\\\" for nom, r in casos]
escribir("t_sens_espec.tex", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & & & & \\multicolumn{3}{c}{Sensibilidad / especificidad} \\\\\n\\cmidrule(l){5-7}\n"
         "Modelo & Exact. & Exact. bal. & $\\kappa$ & Adelantado & Puntual & Retrasado \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# XGB vs RF
x = R("test_xgb_vs_rf.csv")
filas = []
for k, c in ESC:
    r = x[(x.tarea == "reg") & (x.k == k) & (x.conjunto == c)].iloc[0]
    q = x[(x.tarea == "clf") & (x.k == k) & (x.conjunto == c)].iloc[0]
    f = lambda z: f"\\textbf{{{pv(z.p_holm)}}}" if z.significativo_holm else pv(z.p_holm)
    filas.append(NE(k, c) + f" & {n(r.XGB)} / {n(r.RF)} & {f(r)} & {n(q.XGB, 3)} / {n(q.RF, 3)} & {f(q)} \\\\")
escribir("t_xgb_rf.tex", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nEscenario & MAE XGB / RF & $p_{\\text{Holm}}$ & F1-m XGB / RF & $p_{\\text{Holm}}$ \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# Cribado de variables nuevas (validación)
cr = R("cribado_CP_-1_5.csv")
NG = {"+C": "+ C (día en curso)", "+P": "+ P (frecuencia prog.)", "+C+P": "+ C + P", "+A": "+ A (anomalía del día)"}
filas = []
for t, k in (("reg", 0), ("reg", 12), ("clf", 0), ("clf", 12)):
    g = cr[(cr.tarea == t) & (cr.k == k)]
    b = g[g.grupo == "base"].iloc[0]
    d_ = 2 if t == "reg" else 3
    celdas = [n(b.media, d_)] + [f"{n(g[g.grupo == gg].media.iat[0], d_)} ({int(g[g.grupo == gg].pliegues_ganados.iat[0])}/4)" for gg in NG]
    filas.append(f"{'MAE' if t == 'reg' else 'F1-m'}, $k={k}$ & " + " & ".join(celdas) + r" \\")
escribir("t_cribado.tex", "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nMétrica & Base & + C & + P & + C + P & + A \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")

# Sensibilidad de umbrales
filas = []
for u, nom in (("cao", "$-1/+5$"), ("amplio", "$-3/+10$")):
    for k, c in ((0, "F0"), (12, "F1")):
        r = RT.loc[("clf", u, "xgb", k, c)]; m = RT.loc[("clf", u, "B2 histórico", k, c)]
        filas.append(f"{nom} & {k} & {n(r.exactitud, 3)} & {n(m.exactitud, 3)} & {n(r.exactitud_bal, 3)} & {n(r.F1_macro, 3)} & {n(r.kappa, 3)} \\\\")
escribir("t_umbrales.tex", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nUmbrales & $k$ & Exact. & Exact. mayor. & Exact. bal. & F1-m & $\\kappa$ \\\\\n\\midrule\n" + "\n".join(filas) + "\n\\bottomrule\n\\end{tabular}\n")
