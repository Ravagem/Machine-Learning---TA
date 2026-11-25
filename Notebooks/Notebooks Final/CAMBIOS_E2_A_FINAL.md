# Cifras que cambian del Entregable 2 a la entrega final

Todas las cifras "Final" salen de `resultados/` (corrida local, 1 hilo por modelo, malla ampliada de XGBoost).
**Random Forest, Ridge, Logística, B1 y B2 no cambian** (reproducen E2 al último decimal); solo cambia XGBoost.

## Regresión — XGBoost en prueba (MAE en min, R²)

| Escenario | E2 | Final | Hiperparámetros finales |
|---|---|---|---|
| k = 0, F0 | 2,63 (0,75) | **2,65 (0,75)** | prof. 2, tasa 0,03, 800 árboles, subm. 0,8 |
| k = 4, F1 | 2,65 (0,75) | 2,65 (0,75) | prof. 2, tasa 0,03, 800, 0,8 |
| k = 8, F1 | 2,65 (0,76) | 2,64 (0,76) | prof. 3, tasa 0,03, 800, 0,8 |
| k = 12, F1 | 2,38 (0,81) | 2,37 (0,81) | prof. 3, tasa 0,01, 1 600, 0,8 |
| k = 4, F2 | 2,71 (0,75) | 2,68 (0,75) | prof. 2, tasa 0,03, 800, 0,8 |
| k = 8, F2 | 2,69 (0,75) | 2,69 (0,75) | prof. 3, tasa 0,03, 800, 0,8 |
| k = 12, F2 | 2,38 (0,81) | **2,35 (0,81)** | prof. 3, tasa 0,01, 1 600, 0,8 |

- Validación k = 0: 3,14 ± 0,88 → **3,12 ± 0,88** min.
- Reducción frente a B2 (4,35): 40 % → **39 %** (1 − 2,649/4,346 = 39,0 %).
- **Ojo:** el resumen de E2 dice "2,66 → 2,38" y la Sección VI-D dice 2,65; en la versión final es **2,64 → 2,37**.

## Clasificación (−1/+5) — XGBoost en prueba

| Escenario | F1-macro E2 → Final | Recall "retrasado" E2 → Final | PR-AUC E2 → Final |
|---|---|---|---|
| k = 0, F0 | 0,736 → **0,740** | 0,72 → **0,77** | 0,81 → 0,82 |
| k = 4, F1 (mejor por validación: RF) | 0,722 → 0,722 | 0,86 → 0,86 | 0,81 → 0,81 |
| k = 8, F1 | 0,734 → 0,732 | 0,82 → 0,79 | 0,81 → 0,81 |
| k = 12, F1 | 0,751 → 0,751 | 0,80 → 0,83 | 0,83 → 0,83 |
| k = 12, F1 **+ pesos por clase (D4b)** | — → **0,759** | — → 0,83 | — |
| k = 4, F2 | 0,726 → 0,739 | 0,84 → 0,81 | 0,81 → 0,81 |
| k = 8, F2 | 0,739 → 0,736 | 0,80 → 0,82 | 0,80 → 0,80 |
| k = 12, F2 | 0,756 → **0,751** | 0,80 → 0,82 | 0,82 → 0,82 |

**Métricas nuevas (no estaban en el paper), XGBoost, prueba:**

| | Exactitud | Exact. balanceada | Kappa | Sens./Espec. adelantado | Sens./Espec. puntual | Sens./Espec. retrasado |
|---|---|---|---|---|---|---|
| k = 0 | 0,780 | 0,722 | 0,595 | 0,54 / 0,97 | 0,86 / 0,67 | 0,77 / 0,92 |
| k = 12, F1 | 0,790 | 0,736 | 0,617 | 0,53 / 0,98 | 0,86 / 0,70 | 0,83 / 0,91 |
| k = 12, F1 + pesos | 0,790 | 0,752 | 0,624 | 0,59 / 0,96 | 0,84 / 0,73 | 0,83 / 0,91 |
| Clase mayoritaria | 0,592 | 0,333 | 0,000 | 0 / 1 | 1 / 0 | 0 / 1 |

## Pruebas de hipótesis

| | E2 | Final |
|---|---|---|
| H1 (26 comparaciones) | todas p ≤ 0,0014 | todas p ≤ **0,0010** |
| H2 XGB regresión k = 8 → 12 | p = 0,002 | p = 0,0015 |
| H2 XGB regresión k = 0 → 12 | p = 0,042 | p = **0,0043** |
| H2 XGB clasificación k = 8 → 12 | p = 0,27 | p = **0,0499** (ahora significativo, al límite) |
| H3 XGB k = 4, F1 → F2 | empeora, p = 0,001 | 2,65 → 2,68, **p = 0,29 (ya no significativo)** |
| H3 en general | rechazada | rechazada (ningún F2 mejora significativamente) |
| XGB vs RF (nuevo, Holm) | no se probó | iguales en k = 0, 4, 8; XGB mejor en k = 12 (p_Holm = 0,007 con F1; 0,0003 con F2) |

## Otras cifras

- Sábado 01/02: MAE 5,39 → **5,40** min; 28 % del error (sin cambio).
- Importancia (k = 0): hora +4,04 → **+3,97** min; día +1,59 → **+1,60** min.
- Análisis de sensibilidad (−3/+10, nuevo): exactitud XGB k = 0 = 0,83 frente a 0,79 de la clase mayoritaria; kappa 0,60
  (igual que con −1/+5). Con k = 12: exactitud 0,87, F1-macro 0,81.
- Variables del día en curso (C, A) y frecuencia programada (P): rechazadas en validación (resultado negativo).
