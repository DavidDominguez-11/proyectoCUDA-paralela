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

## DEC-016 — Selección de oportunidades del paso 3 (resuelve DEC-003)

- Fecha UTC: 2026-11-17; responsable: Codex por solicitud del usuario.
- Estado: adoptada para la propuesta documental; revisión humana pendiente.
- Contexto: las cinco corridas CPU permiten priorizar por coste observado,
  mientras el usuario limita esta fase a identificar y justificar oportunidades.
- Alternativas: preparación CUDA, PCA/Random Forest, todas las redes o foco CNN-B.
  Se selecciona la matemática del entrenamiento CNN-B: 805.650 s, 71.94% de la
  referencia común; las otras redes son oportunidades secundarias.
- Operaciones: convoluciones, Dense y sus gradientes como principales; ReLU,
  pooling, BatchNormalization, softmax/pérdida, Adam y Dropout como complementarias
  del mismo entrenamiento. No se prescribe cómo implementarlas.
- Preparación de imágenes en CPU conforme al PDF; cualquier traslado futuro de
  augmentation fuera del modelo exige equivalencia, aún pendiente de diseño.
- Se descartan del objetivo actual PCA/RF por menor prioridad y alcance diferente,
  no por imposibilidad de paralelización. No se seleccionan épocas independientes
  ni entrenamiento concurrente de modelos como sustituto de acelerar operaciones.
- Evidencia y límites: [paso 3](oportunidades-paralelizacion.md), análisis original,
  resumen.csv y referencia común EXP-002; no hay perfil por operador ni speedup GPU.
- Consecuencia: se resuelve la selección conceptual DEC-003; stack, diseño,
  memoria, compatibilidad y mediciones GPU siguen pendientes. La equivalencia de
  carga no permite usar el total de todas las fases como baseline de una sola CNN.
- Revisión: si evidencia posterior de hardware, compatibilidad o perfil operativo
  cambia la conveniencia, documentar una decisión nueva sin borrar esta selección.
- REQ-04/12; PROC-11; CAM-004; BIT-006. Sin cambios de código, datos ni EXP-002.

## DEC-017 — Kernels Python explícitos y reparto CUDA (paso 4)

- Estado: propuesta técnica adoptada para DIS-CUDA-001; sin implementar.
- Fecha:2026-11-17 UTC; responsable:Codex, solicitud del usuario.
- Contexto: REQ-05 exige threads/blocks/grids y REQ-06 CUDA Python; CNN-B es el
  objetivo seleccionado. Inventario identifica GTX1650,4GiB,sm_75,driver591.59.
- Alternativas: backend TensorFlow GPU oculta configuración; CUDA C desde Python
  cambia el lenguaje de kernels; numba-cuda permite kernels Python y reparto
  explícito. Se propone NVIDIA numba-cuda con bindings CUDA Python y un stream.
- Diseño: GEMM implícito tiled16×16, dW particionado2048 con reducción posterior,
  vectores/reducciones256threads, softmax32threads, sin atómicos flotantes;
  pooling gather, BN poblacional y Adam fiel al fuente local. FP32, NHWC,
  sin Tensor Cores/TF32/mixed precision; tope768MiB de buffers propios.
- Evidencia: [diseño principal](diseno-paralelo.md), [catálogo](diseno-kernels-cuda.md)
  y [consulta de inventario](../../resultados/diseno/paso-4-inspeccion.md).
- Consecuencia: DEC-004 se resuelve a nivel de stack propuesto; lock y admisión
  binaria siguen por verificar. PCUDA no tiene el stack instalado; no modificar
  sus paquetes para forzar compatibilidad. No se afirma que el algoritmo sea
  óptimo ni que supere CPU sin mediciones. Cambiar backend requiere DEC/CAM nuevos.
- REQ-05/06; PROC-12; CAM-005; BIT-007. Revisión humana pendiente.

## DEC-018 — Equivalencia explícita y CPU pareada nueva

- Estado: contrato adoptado para comparación futura, sin ejecuciones nuevas.
- Fecha:2026-11-17 UTC; responsable:Codex.
- Problema: compartir una seed entre generadores distintos no iguala máscaras
  Dropout ni orden; EXP-002 además contiene varias arquitecturas y ramas adaptativas.
- Se fija CNN-B base15épocas, batch128, último20, sin augmentation ni selección
  por test. Importar estado inicial preservado; orden PCG64/SeedSequence por época
  y máscaras CPU por capa/paso/repetición, comunes en ambos dispositivos.
- Preparación CPU6workers acotados, misma salida exacta; validación y evaluación
  final en CPU para mantener CUDA restringido al entrenamiento conforme al PDF.
  D2H de modelo por época y generación de máscaras se incluyen en el tiempo total.
- Alternativa: usar RNG distinto y comparar históricos directamente; rechazada
  porque no iguala control aleatorio ni alcance. Igualar distribución no reproduce
  trayectoria; conservar como procedencia los cuatro estados RNG del NPZ antiguo.
- Consecuencia: futura implementación de adaptador CPU y nuevos RUN CPU/GPU bajo
  EXP nuevo congelado. EXP-002 queda intacto y solo justifica la prioridad.
- DEC-005 mantiene referencia histórica; DEC-006 se resuelve como6workers
  propuestos y DEC-007 como augmentation excluido del alcance base.
- Evidencia: [contratos y validación](diseno-validacion-cuda.md); REQ-07/12;
  CAM-005; BIT-007. Calidad y repetibilidad quedan pendientes de prueba.

## DEC-019 — Temporización completa y aceptación previa

- Estado: protocolo propuesto, sin tiempos GPU observados.
- Fecha:2026-11-17 UTC; responsable:Codex.
- Se elige un stream sin solapamiento para trazabilidad; tiempo Host sincronizado
  como principal y eventos por grupos con alcance explícito. Publicar copias,
  staging, flags, validaciónCPU y administración, sin sumar intervalos anidados.
- Cinco parejas CPU/GPU, orden alternado fijo y extensión a7/9 si semiancho
  ICt95% supera5%; no seleccionar repeticiones por un resultado favorable.
- Puertas de integridad/entorno/memoria/operadores/pasos/calidad, tolerancias
  numéricas previas y capturas/logs durante carga. Fallos se conservan; no se
  reduce batch, se cambia dtype, desactiva TDR o activa fallbackCPU silencioso.
- Alternativas: contar solo lanzamiento/kernel o esconder copias; no cubren REQ-07.
  Perfilar todo en el timing principal alteraría el coste; profiling separado.
- DEC-008 resuelta como diseño; DEC-009 conserva pendiente la verificación del
  entorno CUDA y lock real. No se confunde inventarioGPU con evidencia de uso.
- Evidencia: [plan de validación/medición](diseno-validacion-cuda.md).
- REQ-05/07/08/09/12; PROC-12; CAM-005; BIT-007.

## DEC-020 — Contraste con Semana15 y coalescencia de reducciones

- Fecha:2026-11-17 UTC; responsable:Codex por solicitud complementaria del usuario.
- Estado: incorporada a DIS-CUDA-001 revisión1, aún sin implementar.
- Fuente complementaria: [copia exacta de clase](../referencias/semana-15-programacion-cuda.md),
  SHA-256 acce03142813c9aca0bf4329000639bb2bb986f92e2810bd1c291774e73fab26.
  Conserva precedencia absoluta del PDF. [Contraste](contraste-semana-15.md).
- Hallazgo: el reparto inicial de reducciones1D por canal leía filas NHWC con
  strideC entre lanes. Se cambia R(L,C) a block32×8, manteniendo256threads y2048
  filas por partición: cada warp lee32canales contiguos; suma local por ty y
  árbol compartido sobre las ocho filas del block. RF final conserva block256.
- Alternativas: mantener1D, transponer tensores o cambiar layout completo. Se
  adopta32×8 para mejorar el patrón de lectura sin buffers grandes adicionales.
  El coste serial por thread y el redondeo deben validarse; no es mejora medida.
- Se mantienen tiles16×16,32threads softmax,control1 y stream único. Las reglas
  256/512de clase se usan como orientación, no como garantía de máxima ocupación.
  Se distinguen SIMT, predicación, memoria local y registros con fuentes NVIDIA.
- Consecuencia: actualizar catálogo/tabla/ejemplos manteniendo presupuesto,
  parámetros, pasos y payload. Ninguna corrida o código de EXP-002 cambia.
- REQ-05/12; PROC-12; CAM-005; BIT-007. Sin instalación ni experimentos.

## DEC-021 — Reutilizar CPU histórica y adaptar entorno disponible

- Fecha:2026-11-17 UTC; responsable:Codex por instrucción posterior del usuario.
- Estado: aplicada como revisión del diseño antes de implementar/medir GPU.
- Sustituye la obligación de CPU pareada y máscaras compartidas de DEC-018/019.
  No invalida las mediciones históricas ni exige repetir otras redes/PCA/RF.
- Misma arquitectura CNN-B, datos/particiones/estado inicial,15épocas,batch128 y
  último20; Dropout con distribución original y seed documentada. Igualdad exacta
  de máscaras se reserva para pruebas pequeñas de corrección.
- Comparación de tiempos solo si fronteras y trabajo coinciden; diferencias
  temporales/térmicas de una referencia histórica se revelan. Si cambia carga,
  no calcular speedup falso ni ejecutar otra campaña CPU sin necesidad concreta.
- PCUDA existente; añadir únicamente dependencias ausentes sin cambiar NumPy,
  TensorFlow u otros paquetes ya instalados. Usar toolkit13.1 disponible en vez
  del12.x propuesto, con smoke real y versiones/hashes archivados.
- CAM-006; BIT-008; [registro del paso5](implementacion-cuda.md).

## DEC-022 — Revisión de productos matriciales por evidencia de admisión

- Estado: implementada antes de congelar mediciones completas. CAM-006; BIT-008.
- Ensayo sintético b128, n=1 por backend, no resultado final: versión tiled
  explícita0.6014s/lote; im2col/SGEMM0.2577s/lote. No se usa como speedup definitivo.
- Se conserva la variante Python tiled como referencia verificable y se usa
  cuBLAS SGEMM FP32 pedantic sin atomics en la serie. Python NumPy nunca multiplica
  los tensores de entrenamiento. Numba ejecuta im2col/col2im gather, no preparación
  de archivos/imágenes; son transformaciones internas de convolución en la GPU.
- Reutilizar workspace global de columnas576MiB, sin retener im2col por capa.
  Nuevo tope buffers1408MiB. La consulta real confirma4GiB; registro de consumo
  por RUN. Los grids internos GEMM pertenecen a NVIDIA: no se inventan16×16.
  Los kernels propios conservan jerarquía explícita documentada.
- Kernels Python numba-cuda0.30.4 requiere alias NumPy retirado row_stack para
  registrar operaciones; alias de proceso row_stack=vstack, sin modificar
  site-packages ni cambiar NumPy2.5.5. Toolkit13.1 y DLLcuBLAS local con SHA-256.
- [Código](../../src/asl_cuda/blas.py); [ensayos](../../resultados/preparacion/cuda/validacion-003/validacion.json).

## DEC-023 — Separar error de operador y sensibilidad de rutas no suaves

- Estado: criterio de admisión fijado antes del entrenamiento medido; no se
  declaran aprobadas las comparaciones estrictas de red completa que fallaron.
- En diagnósticos b2/b20 las activaciones difieren por redondeo FP32. Se observaron
  1 y14rutas de MaxPooling distintas, respectivamente, todas compatibles con
  la cota de redondeo de sus ventanas. El gradiente completo cambia de ruta cerca
  de empates; no es adecuado exigir igualdad de todas sus coordenadas entre
  backends que reducen en órdenes distintos.
- No se aflojan tolerancias de kernels:51pruebas independientes de operadores
  pasan atol1e-4/rtol1e-3 (Adam1e-7/1e-5);78adjuntos de la red con activaciones y
  gradientes de entrada **idénticos** pasan esos límites en b20 y128.
- Diez pasos sintéticos, último20, contador real, exportación, validaciónCPU sin
  mutación, guardado/recarga de pesos+Adam y restauración pasan. Son admisión de
  implementación, no calidad ni timing sobre el dataset real.
- Los diagnósticos fallidos se preservan como tales. Se reportarán curvas y
  accuracy reales, sin afirmar identidad numérica o mejora de calidad debida a
  CUDA. Tampoco se conserva el umbral2pp pareado de un protocolo ya sustituido:
  las propias referencias CPU varían≈6pp y no hay máscaras históricas compartidas.
- Evidencia: operadores-002,adjuntos-001,integracion-002 y rutas-004 bajo
  resultados/preparacion/cuda. La serie real aún exige hashes/datos/pasos completos,
  finitud, checkpoints y medición honesta con validación y transferencias.

## DEC-024 — Análisis del paso 6 sobre evidencias cerradas

- Fecha:2026-11-18UTC. Responsable:Codex por solicitud del usuario;CAM-009;BIT-011.
- Alcance:comparación CNN-B de5CPU EXP-002 y5CUDA EXP-003;sin nuevas ejecuciones
  experimentales,sin informe final y sin exportar PDF.
- Fuentes:CSV originales por RUN,configuraciones y evidencias selladas;reservas
  de revision-pasos-1-5.md. Crear derivados en resultados/comparacion/paso-6,
  conservando todos los artefactos previos y registrando sus hashes.
- Métrica principal:media de pared entrenamiento con transferencias,staging y
  validaciónCPU;speedup=cociente de medias. Diferencia de medias en segundos y
  reducción porcentual se informan aparte. No emparejar artificialmente RUN
  históricos por número ni calcular una media de cocientes pareados.
- Preparación1→6workers se identifica como mejora CPU;test sigue CPU. Suma de
  tres fases comparables se declara derivada;no es todo el notebook ni beneficio
  exclusivo del dispositivo. H2D/D2H no tienen speedup sin fase CPU equivalente.
- Mostrar observaciones individuales y DE muestral en gráficas;ICt95% nominal
  solo con supuestos explícitos,sin inferir significancia o generalización.
- Distinguir medición,cálculo e hipótesis causal. No reinterpretar fallos de
  corrección global,saturaciónGPU ni condiciones históricas como validaciones.
- Guardar Markdown,CSV,PNG/SVG,fuentes,versiones,comando y manifiesto de derivados.
  Estas salidas son análisis posprocesado,por lo que no crean EXP/RUN nuevos.

## DEC-025 — Informe final editable DOCX, sin exportación PDF

- Fecha:2026-11-18UTC. Responsable:Codex,por solicitud explícita del usuario;
  CAM-010;BIT-012. La exportación PDF queda a cargo del usuario y no se ejecuta.
- Portada confirmada por el usuario:David Domínguez23712,Jonathan Diaz23873,
  Luis Padilla23663. Identificar integrantes no prueba aportes individuales/commits.
- Mantener seis apartados principales con los títulos/orden de la guía. Integrar
  datos/procedimiento,tablas de operaciones/repeticiones,CPU,oportunidades,diseño,
  implementación,evidencias de hardware,comparación,speedup y tres preguntas técnicas.
- Evidencia prioritaria:capturas originales CPU/GPU seleccionadas,logs reales y
  monitorización archivada. Distinguir captura de gráfica derivada. No recrear ni
  inventar una captura faltante de CPUhistórica ni certificar saturación literal.
- Informe autónomo,sin rutas/referencias a Markdown internos. Citas externas APA7
  verificadas en fuentes primarias;trazabilidad interna se conserva fuera del DOCX.
- Crear con python-docx ya instalado en Python del sistema,sin cambiar PCUDA.
  Revisar estructura y maquetación con Word local;si se renderiza para inspección,
  usar XPS/PNG y no generar PDF. No repetir experimentos ni cambiar evidencias.

## DEC-026 — Notebooks separados y completos para revisión de ejecución

- Solicitud posterior:entregar únicamente dos notebooks principales,secuencial y
  CUDA,con código por fases y todas las salidas disponibles. CAM-011;BIT-013.
- Usar últimas corridas completas CPU05(EXP-002) yGPU05(EXP-003) como recorrido
  principal;incorporar cinco pruebas completas como registros adicionales.
  No mezclar código del pipelineCPUcompleto dentro del notebookCUDA.
- Desplegar código de los controladores realmente medidos por fase mediante AST,
  preservando semántica de las operaciones y mostrando la separación CPU/GPU.
  Guardar logs íntegros,tablas sin truncar,gráficas y evidencias dentro del ipynb.
- Modo predeterminado:consultar registros originales,celdas de presentación
  ejecutadas realmente conPCUDA. Identificar explícitamente que entrenamientos
  son históricos;no inventar contadores/salidas ni atribuirles ejecución actual.
- Modo opcional de cálculo nuevo requiere activar bandera y carpeta nueva;su
  recorrido inline no se considera medido ni validado experimentalmente por
  haber reconstruido la presentación. Conservar intactos originales yEXPcerrados.
- Se propuso opcionalmente al usuario reutilizar registros o repetir unaCPU/unaGPU.
  Sin respuesta,se continúa con la opción recomendada de reutilización para
  evitar repetir las campañas solo por el empaquetado de salidas.
- Resolución posterior antes de ejecutar:el usuario pidió elegir lo más correcto
  para la revisión detallada del docente. Se decidió y comunicó **ejecutar una
  corrida nueva CPU y unaCUDA directamente en sus notebooks**,con salidas nativas
  junto al código de cada fase. Sustituye la modalidad histórica predeterminada
  descrita arriba. No se repiten las campañas de5ni se recalculan sus promedios.
  NuevosEXP/RUN para estas ejecuciones adicionales;originales/evidencias intactos.

## Estado de decisiones técnicas

| ID | Pregunta | Alternativas a evaluar | Evidencia necesaria | Estado |
|---|---|---|---|---|
| DEC-003 | ¿Qué modelo y operaciones se aceleran? | CNN-B,según DEC-016 | EXP-003 y código aplicado | Resuelta e implementada para CNN-B |
| DEC-004 | ¿Qué stack satisface CUDA Python y diseño explícito? | DEC-017/022:Numba-CUDA Python + cuBLAS | Código,pruebas,recursos propios y límites de backend documentados | Admitido y ejecutado localmente |
| DEC-005 | ¿Cuál será la carga experimental? | DEC-021:CPU histórica compatible,sin otra campaña;GPU5corridas | Configuraciones/series EXP-002/003 | Resuelta con límites explícitos de RNG/referencia histórica |
| DEC-006 | ¿Cómo preparar imágenes con CPU? | 6workers,12pendientes,coordinador escribe orden fijo | X/y idénticos por SHA-256;media8.034s | Implementada y medida,mejora CPU |
| DEC-007 | ¿Cómo tratar augmentation? | DEC-018: fuera del alcance CNN-B base; preparación siempre CPU si se amplía | Separación del PDF y equivalencia futura | Resuelta para DIS-CUDA-001 |
| DEC-008 | ¿Cómo medir transferencias y cómputo? | Pared sincronizada y eventos de grupos,stream único | Tiempos/transferencias por RUN EXP-003 | Medido;eventos no equivalen a tiempo activo puro |
| DEC-009 | ¿Cómo fijar entorno y precisión? | PCUDA,FP32,toolkit13.1,paquetes ausentes añadidos sin cambiar anteriores | hardware.json,contexto,entorno.md y pruebas reales | Admitido localmente;reconstrucción limpia no certificada |

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


## DEC-027 — Ampliación CUDA a FC-A y FC-B

Aceptada por el usuario. Ejecutar exclusivamente implementaciones/pruebas nuevas FC; reutilizar cinco mediciones CPU EXP-002 y CNN-B EXP-003, sin repetirlas. Misma arquitectura, estado inicial histórico, FP32, 15 épocas, batch128 (último20), validación CPU incluida y test CPU separado. Comparación histórica con RNG distinto ya reconocido. FC conserva BN momentum0.99 y dropout0.3/0.4. Código aditivo, memoria dimensionada sin workspace de convoluciones. Cinco RUN nuevos por modelo; EXP-010. Preservar diseño DOCX editado por usuario y salidas previas de notebooks.


Ampliación FC, corrección documental del intento: EXP-010 completó 15épocas FC-A pero falló al exportar CSV de esquema heterogéneo; excluido por cierre incompleto. Se conserva íntegro con traceback,épocas,métricas y captura. EXP-011 corrige únicamente esquema de filas y retoma campaña nueva. No se ejecutó CPU/CNN-B.


La primera corrida FC-A EXP-011 terminó entrenamiento, test, checkpoint y monitor. Solo falló presentación Styler por jinja2 ausente. **Se reutiliza sin repetirla**. EXP-012 ejecutará únicamente nueve pendientes (FC-A2–5,FC-B1–5) y mostrará DataFrame nativo sin dependencias adicionales. Se preservan notebook y error originales.


## Cierre DEC-027

Aceptada e implementada para FC-A y FC-B. Se reutilizaron CPU/CNN-B previas y la primera FC-A ya guardada. Diez corridas completas FC, sin reoptimizar la métrica predictiva. Uso de GPU demostrado con capturas reales; no se fuerza saturación. [Resultados y reservas](ampliacion-fc.md).
