# Predicción de la impuntualidad en una ruta de transporte público de Cusco

Proyecto final del curso **1INF02 Aprendizaje de Máquina — 2026-2** (PUCP).

- **Integrantes:** Equipo

Estudio de aprendizaje automático para predecir el retraso con que un bus de una ruta urbana de Cusco termina su vuelta, a partir de los datos de un sistema de transporte inteligente (horas programadas y reales de paso por 18 puntos de control, y registros GPS de 45 vehículos de enero de 2025).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `AM2026-2_Entregable1_Analisis.ipynb` | Código del **Entregable 1**: reproduce las cifras, tablas y figuras del informe (calidad de datos, ubicación de los puntos de control, patrones de retraso y análisis preliminar de viabilidad). Ya incluye los resultados ejecutados. |
| `AM2026-2_Experimentacion_Plantilla.ipynb` | Plantilla del **Entregable 2**: carga de datos, constantes del diseño experimental y estructura por etapas con celdas `TODO` para implementar la experimentación. |
| `fig/` | Figuras generadas por el notebook del Entregable 1 (se crea al ejecutarlo). |

## Datos

Los datos los proporcionaron los docentes y **no se incluyen en el repositorio**. Deben obtenerse del curso y colocarse en una carpeta propia:

| Archivo | Contenido | Tamaño aproximado |
|---|---|---|
| `Times.csv` | 4 707 vueltas, cada una con 18 puntos de control (listas de horas programadas, horas reales y desviaciones en minutos) | 2 MB |
| `Monitoring.csv` | 1 630 431 registros GPS (fecha, hora, vuelta, latitud, longitud, velocidad, rumbo y vehículo) | 120 MB |

**Aviso:** el README original de los datos describe `difference_time` como `end_time − start_time`. La verificación numérica (Entregable 1, sección IV-B) muestra que `start_time` son las horas **programadas**, `end_time` las **reales**, y `difference_time = trunc((real − programada)/60)` en minutos; un valor negativo indica adelanto. Esta interpretación está pendiente de confirmar con los docentes.

## Cómo ejecutar los notebooks en Google Colab

1. Subir los notebooks a Colab (`Archivo → Subir cuaderno`) o abrirlos desde GitHub.
2. Crear en Google Drive la carpeta `AM2026-2/data/` y copiar allí `Times.csv` y `Monitoring.csv`. Se recomienda esta opción porque `Monitoring.csv` es pesado y subirlo en cada sesión es lento.
3. En la celda de carga, dejar `MODO_CARGA = "drive"` (o cambiarlo a `"upload"` para subir los archivos manualmente) y ajustar `DRIVE_DIR` si usan otra ruta.
4. Ejecutar las celdas en orden (`Entorno → Ejecutar todas`). El notebook del Entregable 1 tarda varios minutos, sobre todo en las secciones 7 y 10.

También puede ejecutarse localmente: definir la variable de entorno `DATA_DIR` con la carpeta de los CSV (por defecto `./data`).

## Dependencias

Python 3.13 con las versiones usadas en el análisis preliminar:

| Librería | Versión |
|---|---|
| pandas | 3.0 |
| numpy | 2.5 |
| scikit-learn | 1.9 |
| xgboost | 3.4 |
| matplotlib | 3.11 |
| scipy | incluida en Colab |

Colab ya las trae instaladas; los notebooks imprimen las versiones al iniciar. La semilla aleatoria es `42`.

## Diseño experimental (resumen)

- **Unidad de análisis:** la vuelta. **Una sola variable objetivo por vuelta:** la desviación en el punto de control 18 (PC18), en minutos.
- **Tareas:** regresión (retraso en minutos) y clasificación en tres clases: adelantado (`< −1` min), puntual (`−1` a `5` min) y retrasado (`> 5` min).
- **Escenarios:** se predice con k = 0 (antes de salir), 4, 8 y 12 puntos de control ya observados.
- **Conjuntos de variables:** F0 (contexto del itinerario), F1 (F0 + desviaciones observadas) y F2 (F1 + agregados GPS).
- **Modelos:** Ridge / Regresión Logística, Random Forest y XGBoost, frente a líneas base de persistencia y promedio histórico.
- **Validación:** partición **temporal**. Prueba: 28/01–01/02/2025 (se usa una sola vez). Desarrollo: 06/01–27/01, con validación cruzada de ventana expansiva de 4 pliegues.
- **Métricas:** MAE (principal), RMSE y R² en regresión; F1-macro (principal), *recall* de "retrasado", matriz de confusión y PR-AUC en clasificación.

Los detalles y su justificación están en el informe del Entregable 1.

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

- **Fuga de información:** las variables del escenario *k* solo pueden usar datos anteriores al paso por el punto de control *k*.
- Las celdas `✅ Verificación` de la plantilla comparan los resultados con estas cifras; si fallan, revisar la implementación antes de continuar.
- El tiempo detenido del GPS debe calcularse como **duración**, no como número de registros: el GPS reporta con menos frecuencia cuando el bus está detenido.
