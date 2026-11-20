# Predicción de la impuntualidad en una ruta de transporte público de Cusco

Proyecto final del curso **1INF02 Aprendizaje de Máquina — 2026-2** (PUCP).

- **Integrantes (Grupo 06):** Jafeth Manases Cuyo Alata, Raul Esteban Herrera Guerrero, Matias Esteban Marrufo Zapata.

Estudio de aprendizaje automático para predecir el retraso con que un bus de una ruta urbana de Cusco termina su vuelta, a partir de los datos de un sistema de transporte inteligente (horas programadas y reales de paso por 18 puntos de control, y registros GPS de 45 vehículos de enero de 2025).

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| `Dataset/` | `Times.csv` y el README original de los datos. `Monitoring.csv` no se incluye (ver [Datos](#datos)). |
| `Notebooks/Notebooks E1/AM2026-2_Entregable1_Analisis.ipynb` | **Entregable 1**: reproduce las cifras, tablas y figuras de la propuesta (calidad de datos, ubicación de los puntos de control, patrones de retraso y análisis preliminar de viabilidad). Incluye los resultados ejecutados. |
| `Notebooks/Notebooks E2/AM2026-2_Experimentacion.ipynb` | **Entregable 2**: experimentación completa (preprocesamiento, EDA, variables F0/F1/F2, partición temporal, líneas base, búsqueda en malla, evaluación en prueba, pruebas estadísticas, ablaciones e importancia de variables). Incluye los resultados ejecutados. |
| `Notebooks/Notebooks E2/resultados/` | Tablas de resultados en CSV generadas por el notebook del Entregable 2 (ver abajo). |
| `Notebooks/Notebooks E2/fig/` | Figuras del Entregable 2; las de prefijo `paper_` son las versiones a ancho de columna usadas en el paper. |
| `Informes/Entregable2_Grupo06.docx` | Paper del **Entregable 2**: implementación, resultados y discusión (formato IEEE). |

## Datos

Los datos los proporcionaron los docentes del curso:

| Archivo | Contenido | Tamaño aproximado | En el repositorio |
|---|---|---|---|
| `Times.csv` | 4 707 vueltas, cada una con 18 puntos de control (listas de horas programadas, horas reales y desviaciones en minutos) | 2 MB | Sí, en `Dataset/` |
| `Monitoring.csv` | 1 630 431 registros GPS (fecha, hora, vuelta, latitud, longitud, velocidad, rumbo y vehículo) | 120 MB | No: supera el límite de 100 MB por archivo de GitHub. Debe obtenerse del curso. |

**Aviso:** el README original de los datos describe `difference_time` como `end_time − start_time`. La verificación numérica (Entregable 1, sección IV-B) muestra que `start_time` son las horas **programadas**, `end_time` las **reales**, y `difference_time = trunc((real − programada)/60)` en minutos; un valor negativo indica adelanto. Esta interpretación está pendiente de confirmar con los docentes.

## Cómo ejecutar los notebooks en Google Colab

1. Abrir el notebook desde GitHub en Colab (`Archivo → Abrir cuaderno → GitHub`) o subirlo (`Archivo → Subir cuaderno`), y guardar una copia en Drive (`Archivo → Guardar una copia en Drive`).
2. Crear en Google Drive la carpeta `AM2026-2/data/` y copiar allí `Times.csv` y `Monitoring.csv`. Se recomienda esta opción porque `Monitoring.csv` es pesado y subirlo en cada sesión es lento.
3. En la celda de carga, dejar `MODO_CARGA = "drive"` (o cambiarlo a `"upload"` para subir los archivos manualmente) y ajustar `DRIVE_DIR` si usan otra ruta.
4. Ejecutar todo (`Entorno de ejecución → Ejecutar todas`).
   - **Entregable 1:** tarda varios minutos, sobre todo en las secciones 7 y 10.
   - **Entregable 2:** tarda unos **30–35 min** en Colab; casi todo el tiempo es la búsqueda en malla (etapa 8). Los resultados y figuras se guardan en `AM2026-2/resultados/` y `AM2026-2/fig/` de Drive.
5. **Si Colab se desconecta** durante el Entregable 2, volver a ejecutar todo: la búsqueda guarda cada combinación en `resultados/resultados_cv_raw.csv` y continúa desde donde quedó. Con ese archivo completo, el notebook entero tarda alrededor de 5 min.

Para probar rápidamente que todo corre, poner `MODO_RAPIDO = True` en la etapa 2 (usa solo el primer valor de cada malla, ~2 min); para el experimento debe quedar en `False`.

También puede ejecutarse localmente: definir la variable de entorno `DATA_DIR` con la carpeta de los CSV (por defecto `./data`).

## Dependencias

Versiones de Google Colab con las que se obtuvieron los resultados del Entregable 2 (el notebook las imprime al iniciar y las guarda en `resultados/versiones.json` y `resultados/requirements.txt`):

| Librería | Versión |
|---|---|
| Python | 3.13 |
| pandas | 2.2.3 |
| numpy | 2.4.6 |
| scikit-learn | 1.6.1 |
| xgboost | 3.4.1 |
| statsmodels | incluida en Colab |
| scipy | incluida en Colab |
| matplotlib | 3.10.0 |

La semilla aleatoria es `42`. Dos ejecuciones independientes con estas versiones dieron las mismas métricas de prueba (diferencia máxima de 1,5 × 10⁻⁶).

## Diseño experimental (resumen)

- **Unidad de análisis:** la vuelta. **Una sola variable objetivo por vuelta:** la desviación en el punto de control 18 (PC18), en minutos.
- **Tareas:** regresión (retraso en minutos) y clasificación en tres clases: adelantado (`< −1` min), puntual (`−1` a `5` min) y retrasado (`> 5` min).
- **Escenarios:** se predice con k = 0 (antes de salir), 4, 8 y 12 puntos de control ya observados.
- **Conjuntos de variables:** F0 (contexto del itinerario), F1 (F0 + desviaciones observadas) y F2 (F1 + seis agregados GPS). En total, 7 combinaciones de escenario y conjunto.
- **Modelos:** Ridge / Regresión Logística, Random Forest y XGBoost, frente a líneas base de persistencia y promedio histórico.
- **Validación:** partición **temporal**. Prueba: 28/01–01/02/2025 (se usa una sola vez). Desarrollo: 06/01–27/01, con validación cruzada de ventana expansiva de 4 pliegues y búsqueda en malla completa (1 239 combinaciones).
- **Métricas:** MAE (principal), RMSE y R² en regresión; F1-macro (principal), *recall* de "retrasado", matriz de confusión y PR-AUC en clasificación. Pruebas pareadas de Wilcoxon y McNemar (α = 0,05).

## Resultados principales (Entregable 2, conjunto de prueba)

| Escenario | MAE XGBoost | MAE promedio histórico | F1-macro XGBoost | F1-macro clase mayoritaria |
|---|---|---|---|---|
| Antes de salir (k = 0) | 2,63 min | 4,35 min | 0,736 | 0,248 |
| Tras 12 PC (k = 12, F1) | 2,38 min | 4,35 min | 0,751 | 0,248 |

- **H1 (el ML supera a las líneas base):** se confirma en las 26 comparaciones (p ≤ 0,0014).
- **H2 (el error baja con k):** solo en parte; la mejora significativa aparece recién de k = 8 a k = 12.
- **H3 (el GPS ayuda):** no se confirma; ningún modelo mejora de forma significativa con F2.
- La hora de salida y el día de la semana dominan la importancia de variables; los sábados las vueltas terminan, en promedio, 10,4 min tarde.

### Archivos de `resultados/`

| Archivo | Contenido |
|---|---|
| `resultados_cv_raw.csv` | Todas las combinaciones de la búsqueda en malla, con la métrica de cada pliegue |
| `resultados_cv_mejores.csv` | Mejor configuración por modelo, tarea y escenario (media ± desviación estándar) |
| `tabla_cv_reg.csv`, `tabla_cv_clf.csv` | Tablas de validación cruzada del paper (Tablas IX y X) |
| `resultados_cv_lineas_base.csv` | Líneas base en validación cruzada |
| `resultados_test.csv` | Métricas en el conjunto de prueba |
| `pruebas_h1_vs_lineas_base.csv` | Wilcoxon y McNemar: mejor modelo frente a las líneas base |
| `ablacion_h2_efecto_k.csv`, `ablacion_h3_gps.csv` | Ablaciones de H2 y H3 |
| `importancia_permutacion.csv` | Importancia por permutación |
| `predicciones_test.csv` | Predicción de cada modelo para cada vuelta de prueba |
| `versiones.json`, `requirements.txt` | Versiones de las librerías |

## Datos depurados: cifras de referencia

Útiles para comprobar que el código reproduce el análisis:

| Concepto | Valor |
|---|---|
| Vueltas tras excluir horarios corruptos | 4 559 |
| Vueltas con retraso final válido | 4 004 (3 246 en desarrollo y 758 en prueba) |
| Retraso final | media 2,72 min, desviación estándar 8,22 min |
| Clases (adelantado / puntual / retrasado) | 17,0 % / 55,3 % / 27,7 % |
| Cobertura del cruce con el GPS | 99,9 % de las vueltas |

## Notas

- **Fuga de información:** las variables del escenario *k* solo usan datos anteriores al paso real por el punto de control *k*.
- Las celdas `✅ Verificación` del notebook del Entregable 2 comparan los resultados con estas cifras y detienen la ejecución si no coinciden.
- El tiempo detenido del GPS se calcula como **duración**, no como número de registros: el GPS reporta con menos frecuencia cuando el bus está detenido.
- Las decisiones de implementación (imputación, escalado, agregados GPS, ponderación de clases, recorrido de la malla) están documentadas en la etapa 2 del notebook del Entregable 2 y en la Sección V del paper.
