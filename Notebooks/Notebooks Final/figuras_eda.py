"""Gráficos exploratorios para la presentación (formato 16:9, letra grande). Salida: fig/pres_*.png"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from prep import cargar, DATA_DIR

AQUI = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(AQUI, "fig")
os.makedirs(FIG, exist_ok=True)

# Paleta validada (references/palette.md del skill de visualización)
AZUL, NARANJA, AQUA, AMARILLO, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
ROJO, TINTA, TINTA2, MUTED, GRID, BASE, SUP = "#e34948", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f3a6a5", "#e34948", "#a32e2d"])
SECUENCIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 19, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 15, "axes.labelcolor": TINTA2, "xtick.color": TINTA2,
    "ytick.color": TINTA2, "axes.edgecolor": BASE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": SUP, "axes.facecolor": SUP, "savefig.facecolor": SUP, "savefig.dpi": 200,
    "legend.frameon": False, "legend.fontsize": 13})
W, H = 12, 6.2


def guardar(fig, nombre, subtitulo=None):
    if subtitulo:
        fig.text(0.012, 0.005, subtitulo, fontsize=11, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if subtitulo else 0, 1, 1))
    fig.savefig(os.path.join(FIG, nombre))
    plt.close(fig)
    print("  ", nombre)


# Coordenadas de los PC (Tabla III del paper)
PC = np.array([(-13.5552, -71.8568), (-13.5535, -71.8606), (-13.5445, -71.8883), (-13.5288, -71.9372), (-13.5216, -71.9651),
               (-13.5256, -71.9833), (-13.5259, -71.9827), (-13.5359, -71.9804), (-13.5441, -71.9821), (-13.5543, -71.9908),
               (-13.5419, -71.9821), (-13.5355, -71.9804), (-13.5226, -71.9818), (-13.5233, -71.9728), (-13.5226, -71.9714),
               (-13.5423, -71.8991), (-13.5441, -71.8888), (-13.5529, -71.8648)])
NOMBRE_PC = {1: "Inicio", 10: "Retorno (oeste)", 13: "Mercado San Pedro", 14: "Av. Tullumayo", 18: "Fin"}


def main():
    d = cargar()
    vm, Dm, vueltas, D, ok = d["vm"], d["Dm"], d["vueltas"], d["D"], d["ok"]
    y = vm.y.values

    # 1. Mapa: retraso medio acumulado en cada PC sobre la traza GPS
    mon = pd.read_csv(os.path.join(DATA_DIR, "Monitoring.csv"), usecols=["latitude", "longitude"]).sample(120000, random_state=0)
    mon = mon[mon.longitude.between(-72.0, -71.85) & mon.latitude.between(-13.565, -13.515)]
    med = np.nanmean(Dm, axis=0)
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)
    fig, axs = plt.subplots(2, 1, figsize=(W, 8.2))
    for ax, (tit, pcs) in zip(axs, (("Ida: PC1 → PC10", range(1, 11)), ("Retorno: PC10 → PC18", range(10, 19)))):
        ax.scatter(mon.longitude, mon.latitude, s=0.3, c=BASE, alpha=0.25, linewidths=0, rasterized=True)
        idx = np.array(list(pcs)) - 1
        ax.plot(PC[idx, 1], PC[idx, 0], color=MUTED, lw=1, ls=":", zorder=2)
        sc = ax.scatter(PC[idx, 1], PC[idx, 0], c=med[idx], cmap=DIVERGENTE, norm=norm, s=170, edgecolors=SUP,
                        linewidths=1.5, zorder=3)
        for j in pcs:
            la, lo = PC[j - 1]
            dx, dy = {6: (-16, 4), 7: (8, -14), 14: (-6, -22), 15: (10, 6), 1: (8, -14), 2: (-6, 10)}.get(j, (8, 6))
            ax.annotate(f"{j}", (lo, la), xytext=(dx, dy), textcoords="offset points", fontsize=12, color=TINTA, weight="bold")
        if 13 in pcs:
            la, lo = PC[12]
            ax.annotate("Mercado San Pedro", (lo, la), xytext=(-150, -40), textcoords="offset points", fontsize=12, color=TINTA2,
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 1})
        ax.set_aspect(1 / np.cos(np.radians(13.54))); ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
        ax.set_title(tit, fontsize=15, fontweight="normal", color=TINTA2)
        ax.set_xlim(-72.0, -71.85); ax.set_ylim(-13.562, -13.515)
    cb = fig.colorbar(sc, ax=axs, orientation="vertical", shrink=0.6, pad=0.01, aspect=25)
    cb.set_label("Desviación media en el PC (min)\nazul: adelanto · rojo: retraso", color=TINTA2)
    cb.outline.set_visible(False)
    fig.suptitle("La ruta: el bus sale adelantado y acumula retraso en el retorno desde el centro",
                 x=0.012, ha="left", fontsize=19, fontweight="bold")
    fig.text(0.012, 0.005, "18 puntos de control (Tabla III del paper). Fondo: 120 000 registros GPS. Media sobre 4 004 vueltas.",
             fontsize=11, color=MUTED)
    fig.savefig(os.path.join(FIG, "pres_01_mapa_ruta.png"), bbox_inches="tight")
    plt.close(fig)
    print("   pres_01_mapa_ruta.png")

    # 2. Perfil del retraso a lo largo de la vuelta por franja horaria
    franjas = [("Madrugada (4–6 h)", 4, 6, AZUL), ("Mañana (6–10 h)", 6, 10, NARANJA), ("Mediodía (10–14 h)", 10, 14, AQUA),
               ("Tarde y noche (14+ h)", 14, 30, AMARILLO)]
    fig, ax = plt.subplots(figsize=(W, H))
    x = np.arange(1, 19)
    for nom, a, b, c in franjas:
        m = (vm.hora_salida >= a) & (vm.hora_salida < b)
        prof = np.nanmean(Dm[m.values], axis=0)
        ax.plot(x, prof, color=c, lw=2.5, marker="o", ms=6, mec=SUP, mew=1.5)
        ax.annotate(f"{nom}  ({m.sum()})", (18, prof[-1]), xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=12.5, color=TINTA)
    ax.axhline(0, color=TINTA2, lw=1); ax.axhline(5, color=MUTED, lw=1, ls=":"); ax.axhline(-1, color=MUTED, lw=1, ls=":")
    ax.text(1, 5.3, "umbral de retraso (+5)", fontsize=11, color=MUTED)
    ax.axvspan(12.5, 14.5, color=ROJO, alpha=0.07, lw=0)
    ax.text(13.5, -3.2, "San Pedro →\nTullumayo", ha="center", fontsize=11, color=TINTA2)
    ax.set_xticks(x); ax.set_xlim(0.6, 22.8)
    ax.set_xlabel("Punto de control"); ax.set_ylabel("Desviación media (min)")
    ax.set_title("El momento del día decide si el bus termina adelantado o retrasado")
    guardar(fig, "pres_02_perfil_franja.png", "Media de la desviación en cada PC por franja de salida programada; entre paréntesis, n.º de vueltas.")

    # 3. Mapa de calor hora × día
    DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    t = vm.assign(h=np.floor(vm.hora_salida).astype(int))
    piv = t.pivot_table(index="dia_semana", columns="h", values="y", aggfunc="mean")
    cnt = t.pivot_table(index="dia_semana", columns="h", values="y", aggfunc="size")
    piv = piv.where(cnt >= 10).dropna(axis=1, how="all")
    fig, ax = plt.subplots(figsize=(W, H * 0.85))
    im = ax.imshow(piv.values, cmap=DIVERGENTE, norm=TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20), aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10.5,
                        color="white" if abs(v) > 11 else TINTA)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(7)); ax.set_yticklabels(DIAS); ax.grid(False)
    ax.set_xlabel("Hora programada de salida"); [s.set_visible(False) for s in ax.spines.values()]
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01); cb.set_label("Retraso final medio (min)", color=TINTA2); cb.outline.set_visible(False)
    ax.set_title("Sábado por la tarde: el patrón más fuerte del retraso")
    guardar(fig, "pres_03_calor_hora_dia.png", "Retraso medio en PC18 por día y hora; celdas con menos de 10 vueltas en blanco.")

    # 4. Retraso medio por fecha (calendario) y partición temporal
    g = vm.groupby("date").agg(media=("y", "mean"), n=("y", "size")).reset_index()
    g["dow"] = pd.to_datetime(g.date).dt.dayofweek
    fig, ax = plt.subplots(figsize=(W, H))
    col = [ROJO if w == 5 else (AZUL if f >= "2025-01-28" else BASE) for f, w in zip(g.date, g.dow)]
    ax.bar(range(len(g)), g.media, color=col, width=0.72)
    ax.axhline(0, color=TINTA2, lw=1)
    i_t = int(np.flatnonzero(g.date.values >= "2025-01-28")[0])
    ax.axvline(i_t - 0.5, color=TINTA, lw=1.5, ls="--")
    ax.set_ylim(-0.5, 16.5)
    ax.text(i_t - 0.2, 16.2, "PRUEBA (5 días)\nse usa una sola vez", fontsize=12, color=TINTA, va="top")
    ax.text(i_t - 0.8, 16.2, "DESARROLLO (22 días)\nvalidación cruzada temporal en 4 pliegues", fontsize=12, color=TINTA, va="top", ha="right")
    for i, r in g.iterrows():
        if r.dow == 5:
            ax.text(i, r.media + 0.3, f"{r.media:+.1f}".replace(".", ","), ha="center", fontsize=12, color=TINTA, weight="bold")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{DIAS[w][0]}\n{int(f[8:])}" for f, w in zip(g.date, g.dow)], fontsize=11)
    ax.set_xlabel("Día (enero 2025 → 1 de febrero)")
    ax.set_ylabel("Retraso final medio (min)")
    ax.set_title("Los cuatro sábados son los días más impuntuales; uno queda en la prueba")
    guardar(fig, "pres_04_calendario_particion.png", "Rojo: sábados. Azul: días de prueba. Gris: desarrollo.")

    # 5. Distribución del objetivo con las clases
    fig, ax = plt.subplots(figsize=(W, H * 0.85))
    yy = np.clip(y, -30, 40)
    bins = np.arange(-30.5, 41.5, 1)
    cls = [(yy < -1, AZUL, "Adelantado (< −1 min)"), ((yy >= -1) & (yy <= 5), BASE, "Puntual (−1 a +5)"), (yy > 5, ROJO, "Retrasado (> +5 min)")]
    for m, c, lab in cls:
        ax.hist(yy[m], bins=bins, color=c, edgecolor=SUP, linewidth=0.6, label=f"{lab}: {100 * m.mean():.1f} %".replace(".", ","))
    ax.legend(loc="upper right")
    ax.set_xlabel("Retraso final en PC18 (min; recortado a [−30, 40])"); ax.set_ylabel("Vueltas")
    ax.set_title("Qué se predice: el retraso final y su clase de puntualidad")
    guardar(fig, "pres_05_distribucion_clases.png",
            f"4 004 vueltas válidas. Media {y.mean():.2f} min".replace(".", ",") + f"; mediana {np.median(y):.0f} min.")


if __name__ == "__main__":
    main()
