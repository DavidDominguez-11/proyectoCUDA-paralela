# Registro de decisiones

Una decisión aceptada explica contexto, alternativas, fundamento, consecuencias,
responsable y evidencias. No marcar como aceptada una opción técnica aún no
evaluada. Al sustituir una decisión, conservar la anterior y enlazar la nueva.

## DEC-001 — Autoridad documental

- Estado: aceptada por instrucción del usuario.
- Fecha: 2026-11-17.
- Contexto: existe un PDF y una transcripción Markdown ampliada.
- Decisión: el PDF es fuente de verdad; las ampliaciones se identifican como
  explicaciones o acuerdos internos.
- Alternativa: tratar ambos archivos como normativos; descartada por precedencia
  explícita del usuario.
- Consecuencia: matriz REQ con páginas del PDF y sin requisitos inventados.
- Evidencia: PDF e inventario-inicial.csv. Relación: REQ-01 a REQ-12.

## DEC-002 — Trazabilidad antes de experimentar

- Estado: aceptada por instrucción del usuario.
- Fecha: 2026-11-17.
- Decisión: documentos informativos en Markdown; resultados detallados en
  Markdown con CSV, logs, configuraciones, imágenes y gráficas como respaldo.
- Alternativa: documentar al final; descartada porque pierde decisiones, fallos
  y condiciones originales.
- Consecuencia: registros separados DAT/PROC/DEC/CAM/EXP/RUN; plantillas con
  pendientes explícitos; resultados históricos identificados.
- Cambio: CAM-001. Evidencia: esta estructura; REQ-12.

## Decisiones técnicas abiertas

| ID | Pregunta | Alternativas a evaluar | Evidencia necesaria | Estado |
|---|---|---|---|---|
| DEC-003 | ¿Qué modelo y operaciones se aceleran? | CNN-A, CNN-B u otro alcance justificado del laboratorio | Coste CPU, representatividad y complejidad | Pendiente |
| DEC-004 | ¿Qué stack satisface CUDA Python y diseño explícito? | APIs CUDA Python / herramientas compatibles; backend de framework con alcance documentado | Cumplimiento REQ-05/06, hardware, compatibilidad y prueba mínima | Pendiente |
| DEC-005 | ¿Cuál será la carga experimental? | Tamaño de muestra, batch, épocas/pasos, repeticiones, warmup, orden | Presupuesto, variabilidad y equivalencia | Pendiente |
| DEC-006 | ¿Cómo preparar imágenes con CPU? | Bucle original, workers de hilos/procesos u otro mecanismo explícito | Tiempo, uso por núcleo, orden determinista y memoria | Pendiente |
| DEC-007 | ¿Cómo tratar augmentation? | Excluir del experimento base o ejecutar preparación en CPU | Comparación equivalente y separación exigida por PDF | Pendiente |
| DEC-008 | ¿Cómo medir transferencias y cómputo? | Eventos/profiler/temporizador sincronizado según stack | Límites claros, solapamiento y sobrecosto | Pendiente |
| DEC-009 | ¿Cómo fijar entorno y precisión? | Plataforma, versiones estables compatibles, dtype, determinismo | Dispositivos efectivos, instalación reproducible y tolerancias | Pendiente |

## Plantilla para resolver una decisión

```markdown
## DEC-NNN — Título
- Estado: propuesta / aceptada / descartada / sustituida por DEC-...
- Fecha UTC y responsable:
- REQ / DAT / PROC / EXP relacionados:
- Contexto y problema:
- Alternativas y criterios de comparación:
- Evidencia consultada (archivo, RUN o fuente exacta):
- Decisión y justificación:
- Consecuencias en corrección, rendimiento, memoria y reproducibilidad:
- Cambios CAM / commits:
- Condiciones para revisarla:
```
