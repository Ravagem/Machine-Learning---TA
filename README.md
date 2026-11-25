# Predicción de la impuntualidad en una ruta de transporte público de Cusco

Proyecto final del curso **1INF02 Aprendizaje de Máquina — 2026-2** (PUCP).

- **Integrantes (Grupo 06):** Jafeth Manases Cuyo Alata, Raul Esteban Herrera Guerrero, Matias Esteban Marrufo Zapata, Joel Cesar Galindo Nuñez.

Estudio de aprendizaje automático para predecir el retraso con que un bus de una ruta urbana de Cusco termina su vuelta, a partir de los datos de un sistema de transporte inteligente (horas programadas y reales de paso por 18 puntos de control, y registros GPS de 45 vehículos de enero de 2025).

## Entrega final

| Qué | Dónde |
|---|---|
| **Paper final** (formato IEEE, LaTeX) | `Informes/Paper_Final/main.tex` → compilar a PDF (ver abajo) |
| **Demostración interactiva** | `Demo/index.html`: abrir con doble clic en cualquier navegador; funciona sin internet |
| **Notebook final** (ejecutado, con todas las salidas) | `Notebooks/Notebooks Final/AM2026_2_Final.ipynb` |
| **Código fuente** (módulos `.py`) | `Notebooks/Notebooks Final/` (tabla siguiente) |
| **Protocolo de experimentos** | `Notebooks/Notebooks Final/PROTOCOLO.md`: reglas de decisión fijadas antes de ver los resultados |
| **Cambios respecto del Entregable 2** | `Notebooks/Notebooks Final/CAMBIOS_E2_A_FINAL.md` |

### Resultados principales (conjunto de prueba, 758 vueltas del 28/01 al 01/02/2025)

| Escenario | MAE XGBoost | MAE promedio histórico | R² | F1-macro XGBoost | Exactitud | Kappa |
|---|---|---|---|---|---|---|
| Antes de salir (k = 0) | **2,65 min** | 4,35 min | 0,75 | **0,740** | 0,780 | 0,595 |
| Tras 12 PC (k = 12, F1) | **2,37 min** | 4,35 min | 0,81 | 0,751 (0,759 con pesos por clase) | 0,790 | 0,617 |

- **H1 (el ML supera a las líneas base):** se confirma en las 26 comparaciones (mayor p = 0,0010).
- **H2 (el error baja con k):** solo en parte; la mejora significativa aparece de k = 8 a k = 12 (2,64 → 2,37 min, p = 0,0015).
- **H3 (el GPS ayuda):** se rechaza; ningún modelo mejora de forma significativa con F2.
- **XGBoost frente a Random Forest** (Holm): equivalentes antes de salir (2,649 vs. 2,650 min); XGBoost es mejor solo con k = 12.
- **Variables del día en curso:** evaluadas con una regla de aceptación fijada de antemano; no mejoran la validación (resultado negativo).
- La hora de salida y el día de la semana dominan la importancia por permutación y los valores SHAP; los sábados las vueltas terminan, en promedio, 10,4 min tarde.

### Código de la entrega final (`Notebooks/Notebooks Final/`)

| Archivo | Qué hace |
|---|---|
| `prep.py` | Limpieza, variables F0/F1/F2 y partición temporal (idénticas al Entregable 2) + grupos de variables del día en curso (C, A, P) |
| `modelos.py` | Modelos, `Pipeline` y métricas (MAE, RMSE, R², F1-macro, exactitud, exactitud balanceada, sensibilidad/especificidad, PR-AUC) |
| `cribado.py` | Cribado de las variables del día en curso (solo validación cruzada) |
| `experimento.py` | Búsqueda en malla con validación cruzada temporal: 4 165 combinaciones (malla ampliada de XGBoost; umbrales −1/+5 y −3/+10) |
| `evaluacion.py` | Selección por validación, evaluación única en prueba, Wilcoxon/McNemar, XGBoost vs. RF con Holm, pesos por clase con validación anidada, importancia por permutación |
| `figuras_eda.py`, `figuras_resultados.py` | Gráficos para la presentación (`fig/pres_*.png`), incluido SHAP |
| `figuras_paper.py`, `generar_tablas.py` | Figuras y tablas LaTeX del paper final (las tablas se generan desde los CSV: ninguna cifra se copia a mano) |
| `construir_notebook.py` | Genera y ejecuta `AM2026_2_Final.ipynb` |
| `resultados/` | CSV de validación, prueba, pruebas estadísticas y predicciones por vuelta |
| `modelos/` | Modelos XGBoost entrenados con el desarrollo (`joblib`) |

### Cómo reproducir la entrega final

1. Python 3.13 e instalar las versiones exactas:

   ```
   pip install -r "Notebooks/Notebooks Final/requirements.txt"
   ```

2. Colocar `Times.csv` y `Monitoring.csv` en `Dataset/` (o definir la variable de entorno `DATA_DIR`).
3. Desde `Notebooks/Notebooks Final/`, en este orden:

   ```
   python prep.py                 # ~10 s; verifica con assert las cifras de E2 (4 004 vueltas, 758 de prueba, pliegues)
   python cribado.py              # cribado de variables del día en curso
   python experimento.py 8        # búsqueda en malla con 8 procesos (~66 min); retoma si se interrumpe
   python evaluacion.py           # ~5 min
   python figuras_eda.py; python figuras_resultados.py; python figuras_paper.py; python generar_tablas.py
   python construir_notebook.py   # genera y ejecuta el notebook final
   ```

4. Demo: desde la raíz del repositorio, `python Demo/construir_demo.py` regenera `Demo/index.html`.

**Reproducibilidad.** Cada modelo se entrena con un solo hilo, por lo que los resultados son deterministas en una misma máquina. XGBoost con submuestreo menor que 1 puede variar unas centésimas entre plataformas o con distinto número de hilos; por eso las cifras de la entrega final difieren levemente de las del Entregable 2 (obtenidas en Colab). Random Forest, los modelos lineales y las líneas base reproducen exactamente el Entregable 2.

### Compilar el paper

En `Informes/Paper_Final/`, ejecutar pdflatex **dos veces** (la primera pasada genera las referencias y la segunda las numera):

```
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Sobre la demostración interactiva

`Demo/index.html` es un único archivo, sin dependencias externas. Tiene cuatro pestañas:

1. **La ruta:** mapa con los 18 puntos de control coloreados por la desviación media, filtrable por franja horaria y sábados.
2. **Explorar una vuelta real:** se elige una vuelta del conjunto de prueba y el número de PC ya observados (k = 0, 4, 8, 12) para ver el retraso y la clase predichos frente a lo real, con la explicación SHAP.
3. **Simulador antes de salir:** retraso esperado según el día, la hora y el horario (predicción promedio sobre la flota).
4. **Resultados:** MAE y F1-macro según k, error por día, matriz de confusión e importancia de variables.

El HTML no ejecuta modelos: todas las cifras se precalcularon con `Demo/construir_demo.py` usando los mismos modelos del paper, y el script verifica que sus predicciones coinciden con las reportadas.

## Contenido completo del repositorio

| Ruta | Descripción |
|---|---|
| `Dataset/` | `Times.csv` y el README original de los datos. `Monitoring.csv` no se incluye (ver [Datos](#datos)). |
| `Notebooks/Notebooks Final/` | **Entrega final** (ver arriba). |
| `Demo/` | Demostración interactiva (`index.html`), su plantilla y el script que la genera. |
| `Informes/Paper_Final/` | Paper final en LaTeX, con `fig/` y `tablas/`. |
| `Notebooks/Notebooks E1/AM2026-2_Entregable1_Analisis.ipynb` | **Entregable 1**: análisis de la propuesta (calidad de datos, ubicación de los PC, patrones de retraso, viabilidad). |
| `Notebooks/Notebooks E2/AM2026_2_Experimentacion.ipynb` | **Entregable 2**: primera ronda de experimentación (Colab). |
| `Notebooks/Notebooks E2/resultados/`, `fig/` | Resultados y figuras del Entregable 2. |
| `Informes/Entregable2_Grupo06.docx` | Paper del Entregable 2. |

## Datos

Los datos los proporcionaron los docentes del curso:

| Archivo | Contenido | Tamaño aproximado | En el repositorio |
|---|---|---|---|
| `Times.csv` | 4 707 vueltas, cada una con 18 puntos de control (listas de horas programadas, horas reales y desviaciones en minutos) | 2 MB | Sí, en `Dataset/` |
| `Monitoring.csv` | 1 630 431 registros GPS (fecha, hora, vuelta, latitud, longitud, velocidad, rumbo y vehículo) | 120 MB | No: supera el límite de 100 MB por archivo de GitHub. Debe obtenerse del curso. |

**Aviso:** el README original de los datos describe `difference_time` como `end_time − start_time`. La verificación numérica muestra que `start_time` son las horas **programadas**, `end_time` las **reales**, y `difference_time = trunc((real − programada)/60)` en minutos (coincide en el 100 % de los 76 980 valores válidos); un valor negativo indica adelanto. Esta interpretación está pendiente de confirmar con los docentes.

## Diseño experimental (resumen)

- **Unidad de análisis:** la vuelta. **Una sola variable objetivo por vuelta:** la desviación en el punto de control 18 (PC18), en minutos.
- **Tareas:** regresión (retraso en minutos) y clasificación en tres clases: adelantado (`< −1` min), puntual (`−1` a `5` min) y retrasado (`> 5` min). Análisis de sensibilidad con `−3/+10` min.
- **Escenarios:** se predice con k = 0 (antes de salir), 4, 8 y 12 puntos de control ya observados.
- **Conjuntos de variables:** F0 (contexto del itinerario), F1 (F0 + desviaciones observadas) y F2 (F1 + seis agregados GPS). En total, 7 combinaciones de escenario y conjunto.
- **Modelos:** Ridge / Regresión Logística, Random Forest y XGBoost, frente a líneas base de persistencia y promedio histórico.
- **Validación:** partición **temporal**. Prueba: 28/01–01/02/2025 (se usa una sola vez; incluye un sábado). Desarrollo: 06/01–27/01, con validación cruzada de ventana expansiva de 4 pliegues.
- **Regla de aceptación de mejoras** (fijada antes de ver resultados): mejorar la media de validación y ganar en al menos 3 de 4 pliegues.
- **Métricas:** MAE (principal), RMSE y R² en regresión; F1-macro (principal), exactitud, exactitud balanceada, kappa, sensibilidad/especificidad por clase y PR-AUC en clasificación. Pruebas pareadas de Wilcoxon y McNemar (α = 0,05), con corrección de Holm en XGBoost vs. RF. Interpretación con importancia por permutación y SHAP.

## Datos depurados: cifras de referencia

| Concepto | Valor |
|---|---|
| Vueltas tras excluir horarios corruptos | 4 559 |
| Vueltas con retraso final válido | 4 004 (3 246 en desarrollo y 758 en prueba) |
| Retraso final | media 2,72 min, desviación estándar 8,22 min |
| Clases (adelantado / puntual / retrasado) | 17,0 % / 55,3 % / 27,7 % |
| Cobertura del cruce con el GPS | 99,9 % de las vueltas |

## Notas

- **Fuga de información:** las variables del escenario *k* solo usan datos anteriores al paso real por el punto de control *k*; las variables del día en curso solo usan vueltas que terminaron antes de la hora programada de salida.
- El tiempo detenido del GPS se calcula como **duración**, no como número de registros: el GPS reporta con menos frecuencia cuando el bus está detenido.
- **Entregable 2 en Colab:** ver el notebook de `Notebooks/Notebooks E2/` (celda de carga con `MODO_CARGA = "drive"` o `"upload"`; ~30–35 min; retoma si Colab se desconecta).
