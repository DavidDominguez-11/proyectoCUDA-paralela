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

## CAM-003 — Instrumentación y medición CPU del paso 2

- Fecha: 2026-11-17. Responsable: Codex; solicitud del usuario para ejecutar CPU.
- Estado: completado con cinco corridas EXP-002; EXP-001 conserva el intento
  interrumpido por el monitor inicial. Resultados en resultados/secuencial/EXP-002.
- REQ-03/07/12; PROC-03/04/05/06; DEC-011/012/013; BIT-005.
- Se añade benchmark_cpu.py: extrae por AST las funciones originales, congela
  selección/particiones, instrumenta fases, lanza procesos nuevos y monitoriza
  CPU/RAM. El notebook permanece intacto.
- Cambios de control experimental: seeds explícitas,15 épocas fijas sin parada
  temprana y número de hilos documentado; se preservan arquitecturas y operaciones.
- Se añade resumir_cpu.py para reconstruir estadísticas, tablas y figuras desde
  tiempos crudos, con intervalos t95%, variabilidad y revisión de equivalencia.
- No hay reinstalación ni descarga. Datos grandes/checkpoints locales en data/,
  ignorados por Git; manifiestos, hashes, configuraciones y evidencia en resultados/.
- Criterio de cierre: corridas completas, hashes/tensores/estados iniciales
  equivalentes, informe y gráficas revisados. Los errores se conservan.
- Resultados observados: referencia común 1119.928 ± 2.723 s (DE muestral,
  n=5); total adaptativo 1858.617 ± 314.366 s. Una selección cambió a CNN-A+aug;
  se estratifican ramas y se documenta el alcance común mediante DEC-014/015.
- El controlador original congelado no se editó después de preparar EXP-002.
  Un supervisor cerró la serie tras cinco salidas correctas; su estado anterior
  se conserva. La extensión06 solo preparó contexto y no produjo una medición.
- Se añadieron utilidades de cierre, sellado y preservación, y una guía/script
  para preparar otra identidad con los mismos manifiestos sin volver a muestrear.
- Auditoría: checkpoints iniciales/finales y manifiestos íntegros; 25 contadores
  Adam de 1440; tensores/paquetes/fotos/estados iniciales coincidentes. La tolerancia
  de accuracy 1e-6 no se alcanzó en CNN-B/variante aumentada; limitación explícita.
- Copia local determinista de 17,400 imágenes en data/EXP-002/fuente-seleccionada.zip,
  fuera de Git, con identidad en preservacion-datos.json/.md. No hubo descarga.
- Cuatro gráficas renderizadas e inspeccionadas. Verificación documental y límites
  en resultados/preparacion/verificacion-preparacion.md y BIT-005.
- No es optimización CUDA ni medición de speedup; no se generó el informe PDF final.


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
