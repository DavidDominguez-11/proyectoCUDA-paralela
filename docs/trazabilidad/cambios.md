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
