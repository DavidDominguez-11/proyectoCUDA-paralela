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

## BIT-004 — Paso 1: análisis de la solución secuencial existente

- Fecha UTC: 2026-11-17. Primera marca temporal explícita de la revisión:
  2026-11-17T16:42:57+00:00. Responsable: asistente Codex; revisión humana pendiente.
- Objetivo: completar las dos tablas del paso 1 y explicar las principales partes
  del programa. REQ-02/12; DAT-01/02/03/04 como antecedentes; PROC-01/02;
  DEC-010; CAM-002. No corresponde EXP/RUN: es una inspección estática.
- Entradas: PDF oficial, notebook original y su inventario; se verificó que el
  notebook mantiene SHA-256
  `ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a`.
- Procedimiento: lectura de la consigna en páginas 1–2 con pypdf; lectura JSON de
  las 68 celdas y revisión de sus 33 celdas de código; identificación de funciones,
  bucles, parámetros y salidas. No se importó ni ejecutó el código del notebook.
- Evidencia durable: [extracción estática](../../resultados/analisis-original/revision-estatica.md),
  SHA-256 `619f9861b51e984b74853fc859587a729a9b92a4cb14602d0470f56bb319f98c`.
- Comando exacto de extracción, ejecutado desde la raíz:

```powershell
python -X utf8 scripts/inspeccionar_original.py --destino resultados/analisis-original/revision-estatica.md
```

- Resultado de extracción: código de salida 0; 33 celdas de código y salidas
  históricas seleccionadas preservadas. El destino se crea sin sobrescribir.
- Resultado documental: cinco partes principales, tabla de 26 operaciones y
  tabla de 25 operaciones repetitivas, con referencias a las celdas y fórmulas.
- Cálculos contrastados: 95×128+20=12,180; 96 lotes/época; suma de épocas visibles
  15+11+15+15+13=69; 6,624 pasos y 840,420 presentaciones train derivadas;
  180,090 presentaciones val; 213,811,200 valores normalizados en el dataset preparado.
- Decisión de alcance: incluir recarga e inferencia, distinguir llamadas de alto
  nivel de operaciones internas, y no suponer referencia monohilo ni CUDA validada.
- Verificación documental desde la raíz:

```powershell
python -X utf8 scripts/verificar_preparacion.py
```

- Salida de verificación: [registro de comprobaciones](../../resultados/preparacion/verificacion-preparacion.md).
  Su alcance es documentación, hashes, esquemas y captura estática, no rendimiento.
- Cambios de algoritmo, datos, hardware o entorno: ninguno. Las cantidades de
  épocas y accuracy son antecedentes, no resultados de nuevas ejecuciones.
- Estado: paso 1 completado por revisión estática. Siguiente paso pendiente:
  diseñar y ejecutar la referencia CPU del paso 2 bajo su protocolo.
- Commit de incorporación: pendiente de realizar por el equipo.

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
