# Protocolo de experimentos de la entrega final (fijado el 24/11/2026, antes de ver resultados nuevos)

Grupo 06 — 1INF02 Aprendizaje de Máquina 2026-2. Este documento fija **qué se prueba y con qué regla se decide**
antes de correr los experimentos, para que ninguna decisión dependa de los números que salgan.

## Se mantiene del Entregable 2 (sin cambios)

- Limpieza (4 707 → 4 559 → 4 004 vueltas), variable objetivo (desviación en PC18) y regla anti-fuga.
- Partición temporal: prueba = 28/01–01/02 (758 vueltas, **incluye el sábado 01/02**), usada una sola vez;
  desarrollo = 06/01–27/01 con los mismos 4 pliegues de ventana expansiva.
- Modelos (Ridge/Logística, Random Forest, XGBoost), líneas base (B1, B2) y escenarios (k = 0, 4, 8, 12; F0/F1/F2).
- Selección de hiperparámetros por la media de validación (MAE o F1-macro). **La prueba nunca decide nada.**
- Los resultados del Entregable 2 se conservan como referencia: la entrega final los reporta junto con los nuevos.

## Regla de aceptación de una mejora (D1)

Una mejora se acepta solo si, en validación cruzada:
1. mejora la métrica principal media (MAE menor o F1-macro mayor), **y**
2. gana en al menos 3 de los 4 pliegues.

Motivo: con pliegues de 377 a 505 vueltas, diferencias pequeñas en la media pueden ser ruido; se exige consistencia.

## Mejoras evaluadas

| Id | Mejora | Cómo se evalúa |
|---|---|---|
| D2 | **Grupo C, contexto del día en curso**: retraso medio de las vueltas del día ya terminadas, media de las 5 últimas en terminar, n.º de vueltas terminadas y retraso final de la vuelta anterior del mismo vehículo. **Corte anti-fuga:** solo vueltas cuya llegada real a PC18 es anterior a la hora programada de salida de la vuelta a predecir. | Cribado (D3) |
| D7 | **Grupo P, planificación**: n.º de salidas programadas en los 60 min previos y minutos desde la salida programada anterior. Se conoce antes del día (itinerario). | Cribado (D3) |
| D3 | **Cribado**: XGBoost con los hiperparámetros elegidos en E2, en k = 0 (F0) y k = 12 (F1), regresión y clasificación, comparando base, +C, +P y +C+P con la regla D1. Solo los grupos aceptados pasan a la corrida completa. | — |
| D5 | **Malla ampliada de XGBoost** (el óptimo de E2 cayó en el borde): se agregan profundidad 2, tasa 0,01 y 1 600 árboles a la malla original. Se reporta como ampliación motivada por el borde. | Corrida completa |
| D4 | **Clases**: se mantienen los umbrales −1/+5 (Cao et al.; rango "1 min antes – 5 min después" usado en la práctica del transporte público). Se agregan exactitud, exactitud balanceada y sensibilidad/especificidad por clase (uno contra el resto). Como **análisis de sensibilidad** se repite la clasificación con −3/+10, reportando siempre la exactitud de la clase mayoritaria como referencia. | Corrida completa |
| D4b | **Regla de decisión para "adelantado"**: pesos multiplicativos sobre las probabilidades de clase, elegidos maximizando el F1-macro sobre las predicciones fuera de muestra de los 4 pliegues. Se acepta con la regla D1. | Tras la corrida completa |
| D6 | **XGBoost frente a Random Forest**: Wilcoxon (regresión) y McNemar (clasificación) en prueba, con corrección de Holm por los 7 escenarios. | Tras la corrida completa |

## Registro de decisiones tomadas durante la ejecución

- **24/11, cribado 1:** C, P y C+P no pasan D1 (empeoran el MAE de validación entre 0,02 y 0,06 min).
  Diagnóstico: el retraso medio de las vueltas ya terminadas está confundido con la hora (antes de las 8 h solo
  terminaron vueltas de madrugada, que van adelantadas; media de C = −4,3 min).
- **24/11, segundo intento declarado (único):** grupo **A, anomalía del día** = retraso de las vueltas terminadas menos
  el promedio histórico de su hora calculado solo con días anteriores. Se evalúa con la misma regla D1. Si no pasa,
  C/A/P se reportan como resultado negativo y no se prueban más variantes.
- **Reproducibilidad:** XGBoost con submuestreo < 1 da resultados levemente distintos según el número de hilos y la
  plataforma (hasta 0,03 min por pliegue; ~0,01 min en la media). Por eso toda la corrida final (incluidos los modelos
  "base") se ejecuta completa en una sola máquina, y las mejoras se juzgan con la regla D1, no con diferencias mínimas.

- **24/11, corrida completa (4 165 combinaciones):** Random Forest, Ridge, Logística y las líneas base reproducen E2 al
  último decimal; solo XGBoost cambia (plataforma + malla ampliada).
- **D5, malla ampliada:** se adopta, porque la regla general de E2 es elegir hiperparámetros por la media de validación y la
  malla ampliada contiene a la original. Aplicando D1 por escenario: pasa en regresión k = 0 (4/4 pliegues), k = 4 F1 (4/4),
  k = 4 F2 (3/4), k = 12 F1 (3/4) y clasificación k = 12 F1 (3/4); en el resto queda igual o empata (2/4). En regresión con
  k = 0 y k = 4 eligió **profundidad 2**: bastan interacciones de dos variables (hora × día).
- **D4b, pesos por clase:** solo pasa D1 en k = 12 con F1 (pesos 1,7 adelantado / 1,0 retrasado; 3/4 pliegues). Se adopta
  solo ahí. En prueba no es significativo (McNemar p = 1,0).
- **D6:** con corrección de Holm, XGBoost supera a Random Forest solo con k = 12 (y en clasificación con k = 4 F1);
  antes de la salida son equivalentes (2,649 vs 2,650 min).

## Qué se reporta

Todo resultado nuevo se presenta junto al del Entregable 2. Si una mejora no pasa la regla D1, se reporta como
resultado negativo o trabajo futuro, no se esconde.
