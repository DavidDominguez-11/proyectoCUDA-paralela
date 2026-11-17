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

## DEC-010 — Alcance y conteo del análisis original (paso 1)

- Estado: aplicada para esta revisión documental, conforme al paso solicitado.
- Fecha: 2026-11-17. Responsable: asistente Codex; revisión humana pendiente.
- Contexto: el notebook incluye entrenamiento, recarga e inferencia, contiene
  paralelismo interno y conserva salidas de diferentes sesiones.
- Alternativas: describir únicamente inferencia o una sola CNN; describir el
  programa completo. Se adopta la segunda para cubrir la solución existente.
- Decisión: usar índices de celda desde cero, preservar el original y distinguir
  cantidades configuradas, fórmulas derivadas y épocas visibles históricamente.
  Documentar entrenamiento completo y el recorrido de inferencia de fotos propias.
- Fundamento: PDF páginas 1–2, REQ-02; lectura del código y salidas guardadas.
- Consecuencia: las cantidades no se presentan como mediciones nuevas; no se
  supone que «secuencial» significa monohilo ni que se haya validado CUDA.
- Evidencia: [análisis original](analisis-original.md) y
  [extracción estática](../../resultados/analisis-original/revision-estatica.md).
- PROC-01/02; CAM-002; BIT-004. No se selecciona todavía modelo ni stack CUDA.

## DEC-011 — Carga CPU completa y repeticiones del paso 2

- Estado: aplicada al diseño de EXP-001 antes de observar tiempos.
- Fecha: 2026-11-17. Responsable: Codex por instrucción del usuario.
- Alcance: funciones originales de preparación y construcción de CNN-A, CNN-B,
  FC-A, FC-B; aumento de la CNN seleccionada, PCA, dos Random Forest, evaluación
  e inferencia en fotos propias. Se excluyen descarga, EDA, renderizado de figuras,
  guardado y recarga redundante de modelos del intervalo de procesamiento.
- Se mantienen 600 imágenes/clase, 64×64 RGB, división 70/15/15 y batch128.
- Cambio controlado: 15 épocas fijas por red sin EarlyStopping; seeds explícitas
  y threads matemáticos fijos. Mantiene arquitecturas, pérdidas y optimizadores.
- Cinco procesos nuevos medidos. Si al terminar n=5 o n=7 el semiancho relativo
  del intervalo t bilateral95% de la media del pipeline supera5%, ampliar a7 o9.
  Al llegar a9 se informa el resultado aunque la precisión objetivo no se cumpla.
  Este criterio de precisión es interno; el IC presupone repeticiones comparables
  aproximadamente independientes y no detecta por sí solo deriva temporal.
- Revisión posterior: DEC-015 suspendió la extensión al observar cargas distintas
  y conservó cinco corridas completas, con referencia común y ramas separadas.
- No descartar outliers de timing. Conservar errores; un proceso fallido detiene
  la serie y requiere diagnóstico explícito antes de reanudar.
- Alternativas: medir solo inferencia o solo una CNN; no cubren el alcance
  completo elegido. Repetir con EarlyStopping cambia el trabajo y dificulta
  comparar CPU/GPU, por eso se fija la longitud.
- PROC-03/04/05/06; REQ-03/07/12; CAM-003; BIT-005.

## DEC-012 — Usar PCUDA y caché existente

- Estado: verificado y aplicado para CPU; no valida GPU.
- Entorno: PCUDA, Python 3.14.3, TensorFlow 2.22.0rc0 ya instalado; no reinstalar.
- Cache: `C:\Users\David\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1`.
  Se verificaron carpetas,29 clases×3,000 imágenes y 28 imágenes oficiales.
- CPU observado: AMD Ryzen5 4600H,6 núcleos físicos/12 lógicos; RAM total
  16,558,346,240 bytes según psutil. TensorFlow detectó CPU y ninguna GPU.
- Forzar CUDA_VISIBLE_DEVICES=-1 y visibilidad GPU vacía; TensorFlow intra-op6,
  inter-op1 y BLAS6; Random Forest conserva n_jobs=-1 del original.
- Preparación conserva el bucle original de un worker. Esta referencia no se
  describe como monohilo: las librerías matemáticas utilizan varios hilos.
- Hash/validación de todos los archivos seleccionados antes de cronometrar.
  Registrar duplicados sin corregir silenciosamente el dataset original.
- Caché de archivos del SO sin vaciar: no se reportará rendimiento de disco frío.
- Se documenta la consulta CIM inicialmente denegada en sandbox y su repetición
  autorizada para identificar CPU; no cambió dependencias ni archivos del dataset.

## DEC-013 — Calentamiento y fronteras temporales

- Estado: definido antes de medir EXP-001.
- Calentar cada arquitectura candidata con148 imágenes train y128val, una época;
  restaurar todas las variables del modelo/optimizador y reiniciar su semilla.
  Conservar checkpoints iniciales, fingerprints y comprobación de restauración.
- La CNN aumentada depende del ganador; calentar ambas posibilidades antes del
  intervalo total y entrenar únicamente la seleccionada por el programa original.
- Medir con perf_counter_ns: preparación, fit por modelo, evaluación, PCA,
  Random Forest e inferencia. La validación dentro de fit se registra como
  subintervalo anidado; no sumarla dos veces.
- pipeline_total es tiempo de pared continuo desde preparación medida hasta el
  final de inferencia propia. Incluye administración/logging interno durante ese
  intervalo; excluye imports, construcción/calentamiento, hashes y checkpoints.
- Muestrear CPU por núcleo, proceso y RAM con psutil desde el controlador cada
  segundo. Misma monitorización en todas las corridas; su coste no se sustrae.
- No se han medido transferencias GPU ni speedup en este paso.

## DEC-014 — Separar equivalencia de carga y repetibilidad del aprendizaje

- Estado: criterio de auditoría aplicado tras observar las dos primeras corridas;
  no cambia código ejecutado, datos, tiempos, seeds ni regla de repeticiones.
- Observación: estados iniciales, paquetes y tensores coinciden, pero CNN-B y su
  variante aumentada no cumplen tolerancia de accuracy 1e-6 entre corridas.
- Se conserva la variación y se declara que la trayectoria no está reproducida
  numéricamente. No se atribuye una causa sin diagnóstico ni se excluye por accuracy.
- Para agrupar tiempos se verifican formas/cantidad de pasos y épocas, selección
  de CNN a aumentar, bosque ganador, fotos y arquitectura efectiva de inferencia.
- Si el ganador final cambia entre una CNN y su versión aumentada, se informa
  el cambio: en inferencia las capas aleatorias están inactivas y la arquitectura
  matemática base coincide. No se exige identidad del nombre para declarar esa
  equivalencia; el nombre exacto permanece en cada RUN.
- Si cambia la arquitectura aumentada o la arquitectura efectiva final, no se
  agrupan esas cargas como equivalentes; se debe estratificar o redefinir el
  experimento de forma explícita, preservando todas las corridas.
- Esta decisión no modificó la regla temporal; la revisión posterior de alcance
  y cierre se registra en DEC-015. Las observaciones originales se conservaron.
- REQ-03/07/12; BIT-005; verificacion-equivalencia.json del análisis final.

## DEC-015 — Estratificar ramas adaptativas y medir una referencia común

- Fecha: 2026-11-17. Adoptada al observar la cuarta corrida, antes del resultado
  de la quinta. No cambia datos, fuente de ejecución, seeds ni observaciones.
- Hallazgo: la cuarta corrida seleccionó CNN-A como mejor CNN y entrena CNN-A+aug,
  mientras las tres primeras entrenaron CNN-B+aug.15 épocas no fijan el mismo
  trabajo si cambia la arquitectura. Esta selección es una condición del programa.
- Se completan cinco corridas del programa y se conservan todos sus tiempos.
  No se excluye la rama rápida ni se promedian variantes como una fase homogénea.
- Referencia CPU fija: suma de 14 intervalos de pared no solapados: preparación;
  fit/evaluate de CNN-A, CNN-B, FC-A y FC-B (8); PCA; fit/validación de ambos
  bosques (4). La validación anidada de cada red ya está dentro de fit.
- Se excluyen de esa suma el diagnóstico de la CNN ganadora, evaluación del bosque
  ganador, aumento y su evaluación, e inferencia del registro adaptativo: dependen
  de selecciones. Se publican por separado; no se borran del total original.
- La referencia común es una suma derivada de intervalos medidos, no un intervalo
  continuo ni tiempo exclusivo de kernels. Todos los modelos base procesan los
  mismos datos/pasos. Se evalúa su precisión estadística con cinco repeticiones.
- El pipeline_total continuo se informa descriptivamente como programa adaptativo,
  con ramas etiquetadas y CI nominal; no se declara referencia de carga fija GPU.
- La extensión automática a 7/9 buscaba precisión del total bajo homogeneidad; esa
  premisa falla. Se detiene el controlador tras cinco corridas correctas mediante
  cerrar_serie_cpu.py y se conserva su estado anterior. Se informa expresamente
  esta revisión y no se declara cumplida la regla original del total mezclado.
- Las fases con una sola observación muestran n=1 y DE/IC no aplicables, nunca 0.
- REQ-03/07/12; PROC-06/09; CAM-003; BIT-005. La selección final para CUDA sigue
  pendiente del paso 3; estas observaciones no prueban speedup.

## Decisiones técnicas abiertas

| ID | Pregunta | Alternativas a evaluar | Evidencia necesaria | Estado |
|---|---|---|---|---|
| DEC-003 | ¿Qué modelo y operaciones se aceleran? | CNN-A, CNN-B u otro alcance justificado del laboratorio | Coste CPU, representatividad y complejidad | Pendiente |
| DEC-004 | ¿Qué stack satisface CUDA Python y diseño explícito? | APIs CUDA Python / herramientas compatibles; backend de framework con alcance documentado | Cumplimiento REQ-05/06, hardware, compatibilidad y prueba mínima | Pendiente |
| DEC-005 | ¿Cuál será la carga experimental? | Tamaño de muestra, batch, épocas/pasos, repeticiones, warmup, orden | DEC-011/013, configuración y serie EXP-002 | Resuelta para CPU; equivalencia GPU pendiente |
| DEC-006 | ¿Cómo preparar imágenes con CPU? | Bucle original, workers de hilos/procesos u otro mecanismo explícito | Tiempo, uso por núcleo, orden determinista y memoria | Pendiente |
| DEC-007 | ¿Cómo tratar augmentation? | Excluir del experimento base o ejecutar preparación en CPU | Comparación equivalente y separación exigida por PDF | Pendiente |
| DEC-008 | ¿Cómo medir transferencias y cómputo? | Eventos/profiler/temporizador sincronizado según stack | Límites claros, solapamiento y sobrecosto | Pendiente |
| DEC-009 | ¿Cómo fijar entorno y precisión? | Plataforma, versiones compatibles, dtype, determinismo | DEC-012: PCUDA existente; hardware.json y contexto de cada RUN | Fijada para CPU; reconstrucción y GPU pendientes |

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
