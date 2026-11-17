# Registro de cambios

## CAM-001 — Preparación de trazabilidad y reproducibilidad

- Fecha: 2026-11-17.
- Origen: solicitud del usuario; DEC-001/002; REQ-12.
- Responsable de preparación: asistente Codex. Revisión y responsables humanos: pendientes.
- Antes: notebook del laboratorio, README mínimos y consigna; sin estructura
  común para decisiones, datos, experimentos, resultados y evidencias.
- Después: documentación Markdown, matriz REQ, análisis original, protocolos,
  plantillas, esquema de configuración, inventario SHA-256 y captura de contexto.
- Continuidad: AGENTS.md indica cómo mantener los registros en trabajos futuros;
  .gitignore permite versionar logs dentro de resultados/.
- Código del algoritmo/datos/modelos: no modificado. No se han ejecutado entrenamientos.
- Impacto en rendimiento/calidad del modelo: no medido; no aplica a este cambio documental.
- Verificación: enlaces locales, JSON/CSV, identidad del original y prueba de
  captura de contexto en directorio temporal; detalle en BIT-003.
- Commit de incorporación: pendiente de realizar por el equipo.

## CAM-002 — Completar paso 1: análisis del original

- Fecha: 2026-11-17. Responsable: asistente Codex; revisión humana pendiente.
- Origen: solicitud de ejecutar exclusivamente el paso 1; REQ-02/12; DEC-010.
- Antes: inventario inicial de operaciones con cuatro columnas, sin el formato
  exacto solicitado ni conteo detallado del resto de bucles y rutas de inferencia.
- Después: explicación de cinco partes principales; tabla numerada de 26
  operaciones y tabla de 25 operaciones repetitivas con datos y frecuencias;
  fórmulas, alcance de recarga y épocas históricas identificados.
- Evidencia añadida: extracción Markdown de las 33 celdas de código y salidas
  seleccionadas; script reproducible `scripts/inspeccionar_original.py` que lee
  JSON y rechaza sobrescribir su archivo de destino.
- Documentos relacionados: análisis, requisitos, decisiones, bitácora, índice
  documental y catálogo de resultados. Original, datos y modelo sin cambios.
- Verificación: extracción estática, comprobación del inventario SHA-256,
  revisión de cálculos derivados y comprobaciones documentales; BIT-004.
- Impacto: completa el análisis del paso 1; no mide tiempos ni resuelve la
  selección de operaciones CUDA del paso 3. No crea corridas experimentales.
- Commit de incorporación: pendiente de realizar por el equipo.

## Plantilla de cambio

```markdown
## CAM-NNN — Título
- Fecha UTC, autor y revisor:
- Motivo / REQ / DEC / incidencia:
- Archivos y funciones afectadas:
- Comportamiento anterior → nuevo:
- Datos, entorno o protocolo afectados:
- Impacto esperado y riesgos:
- Validación realizada y evidencia:
- Impacto observado (RUN y métricas, o pendiente):
- Comparabilidad con resultados anteriores:
- Commit / PR / cambio al que sustituye:
```

Un cambio de optimización requiere demostrar corrección, además de rendimiento.
Un cambio de medición o datos puede invalidar comparaciones y exige explicarlo.
No reescribir retrospectivamente los registros de corridas anteriores.
