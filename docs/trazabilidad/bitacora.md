# Bitácora de trabajo

## Equipo

Los identificadores de fotos no confirman nombres completos ni responsabilidades.
Completar estos campos con los integrantes reales.

| Integrante | Usuario Git | Responsabilidades | Contribuciones verificables |
|---|---|---|---|
| PENDIENTE 1 | PENDIENTE | PENDIENTE | PENDIENTE |
| PENDIENTE 2 | PENDIENTE | PENDIENTE | PENDIENTE |
| PENDIENTE 3 | PENDIENTE | PENDIENTE | PENDIENTE |

## BIT-001 — Inspección inicial

- Fecha de registro: 2026-11-17; actividad previa a esta preparación.
- Responsable: asistente Codex; revisión humana pendiente.
- Procedimientos: PROC-01/02. Requisitos: REQ-02/12.
- Entradas: PDF de tres páginas, notebook de 68 celdas, documentación,
  requirements, Git y metadatos de fotos/entorno.
- Procedimiento: lectura y extracción local del PDF, lectura JSON del notebook,
  inspección de fuentes/salidas guardadas y metadatos. Sin entrenamientos.
- Resultado: análisis original y brechas CPU/CUDA identificadas.
- Evidencia: [análisis original](analisis-original.md); archivos fuente originales.
- Limitación: no existía aún registro durable de cada comando de esa inspección;
  esta entrada resume su alcance y no reconstruye logs inexistentes.

## BIT-002 — Preparar documentación y registros

- Fecha: 2026-11-17.
- Responsable: asistente Codex; equipo/revisión humana pendientes.
- Objetivo: dejar preparado el registro completo antes de implementar y medir.
- PROC-01/02/05; DEC-001/002; CAM-001; REQ-12.
- Entradas: consigna, análisis estático y solicitud del usuario.
- Actividad: creación de documentos, plantillas de RUN, configuración de
  experimento, esquemas CSV y herramienta de captura de contexto.
- Salida: [índice documental](README.md) y [resultados](../../resultados/README.md).
- Evidencia de entradas: [inventario inicial](../../resultados/inventario-inicial.csv).
- Estado: estructura preparada; parámetros técnicos abiertos en decisiones.md.
- Siguiente: seleccionar entorno/modelo y resolver DEC antes de medir.

## BIT-003 — Verificación de la preparación

- Fecha: 2026-11-17. Responsable: asistente Codex.
- Alcance: consistencia de la documentación y funcionamiento de captura estática.
- Comprobaciones: enlaces locales; configuración JSON; cabeceras CSV; hashes del
  inventario; captura en carpeta temporal; validación del ZIP y rechazo de destino
  existente. Las pruebas no ejecutan el algoritmo ni acceden a GPU/dataset remoto.
- Registro: [verificacion-preparacion.md](../../resultados/preparacion/verificacion-preparacion.md).
- Estado definitivo y salida: consultar ese registro; no usarlo como evidencia CPU/GPU.

## Plantilla para cada actividad

```markdown
## BIT-NNN — Actividad
- Inicio / fin UTC:
- Responsable y colaboradores:
- Objetivo / REQ / DAT / PROC / DEC / CAM / EXP / RUN:
- Entradas con versión y hashes:
- Pasos, comandos exactos y directorio:
- Observaciones, errores y resultado:
- Artefactos / evidencia:
- Decisiones derivadas y pendientes:
- Siguiente paso:
- Commit / revisión:
```
