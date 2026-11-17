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

## BIT-005 — Medición secuencial CPU (paso 2)

- Fecha de apertura UTC: 2026-11-17. Responsable: Codex por instrucción del usuario.
- Estado: completada con cinco corridas EXP-002 y alcance revisado DEC-015.
  Los apartados siguientes conservan la secuencia histórica, incluido EXP-001.
- REQ-03/07/12; DAT-01-v1/03-EXP-001/04/05; PROC-03/04/05/06;
  DEC-011/012/013; CAM-003; EXP-001.
- Entorno PCUDA verificado: Python 3.14.3, TensorFlow 2.22.0rc0, CPU visible,
  sin GPU visible. No se reinstalaron dependencias.
- Caché indicada por el usuario verificada:87,000 imágenes en 29 clases y 28 oficiales.
  La selección congelada tiene 17,400 archivos; hashes verificados, sin grupos
  de contenido duplicado dentro de la selección. No se descargó el dataset.
- Incidente de preparación: CIM denegó acceso dentro del sandbox; consulta de
  CPU repetida con elevación autorizada. No afectó datos ni tiempos experimentales.
- Preparación: lectura de protocolos; revisión de APIs del entorno instalado;
  selección explícita de carga/repeticiones antes de medir; creación de
  manifiestos/configuración; validación sintáctica de los scripts.
- Comandos desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cpu.py prepare --dataset "C:\Users\David\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cpu.py run
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cpu.py
```

- El comando prepare terminó con código 0. Configuración original congelada
  SHA-256: `c05dac5f7cf90104f6db2f68601f9a054eee3610a6e36b837b04dd8aa7350d74`.
- Al abrir el registro, los comandos run/análisis estaban pendientes; sus
  resultados efectivos y revisiones se documentan en el cierre de esta entrada.
- Artefactos: resultados/secuencial/EXP-001 y configuraciones/EXP-001.json.
- Warmup, imports y guardado fuera del total; validación de fit anidada;
  no mezclar con tiempos históricos ni declarar speedup GPU.
- Siguiente previsto al abrir: ejecutar serie, revisar variabilidad e informe.

### Incidente y corrección de BIT-005

- EXP-001/RUN-20261117T171504Z-cpu-01 se interrumpió: el monitor mostraba
  RSS de unos 4 MiB y CPU 0% mientras se entrenaba. Se comprobó que PCUDA utiliza
  un lanzador Windows con otro intérprete hijo; el monitor estaba siguiendo solo
  el lanzador. No se usará esa corrida parcial en estadísticas.
- Se conservaron stdout/stderr, recursos, tiempos parciales, salida no cero y
  `incidencia-monitor.md`. El controlador registró EXP-001 como fallida.
- Corrección: monitorizar el árbol completo de procesos, conservar PIDs y sumar
  su CPU/RSS. El RSS agregado puede contar páginas compartidas más de una vez;
  esa limitación se declara, no se confunde con memoria privada exclusiva.
- Comprobación independiente de la corrección: intérprete PCUDA hijo que
  reserva 24 MiB y consume CPU durante 3 s; se detectaron dos PIDs,97.6% CPU y
  40,837,120 bytes RSS agregados. No es timing del algoritmo ASL.
- La corrección genera EXP-002, sin sobrescribir EXP-001. Mantiene carga, semillas,
  particiones y política de repeticiones. Nuevo hash de configuración congelada:
  `6e6ffcb7deacc075bbb30224bd6503621cbe83b36297140fda4ff83ad9a60070`.
- Comandos efectivos de la nueva serie desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cpu.py prepare --experimento EXP-002 --dataset "C:\Users\David\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cpu.py run --experimento EXP-002
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cpu.py --experimento EXP-002
```

- Preparación EXP-002 completada con código 0; la continuación y cierre constan
  en los apartados cronológicos siguientes.

### Primeras dos corridas completas de EXP-002

- RUN-20261117T171942Z-cpu-01 y RUN-20261117T175407Z-cpu-02 terminaron con
  código 0; cada una completó 75 épocas. Media provisional del pipeline:
  2,006.823s; semiancho relativo IC t95%:2.7046%. No es el cierre de la serie.
- Coinciden los hashes de estados iniciales de las seis candidatas, los tensores
  y el inventario de paquetes. Ambas seleccionaron CNN-B, RF-B y CNN-B como
  mejor CNN, bosque y modelo final, respectivamente.
- Se observó diferencia de accuracy de CNN-B (0.96168584 frente a0.98888886)
  y CNN-B+aug (0.95325673 frente a0.86743295). CNN-A, FC-A, FC-B y RF-B
  conservaron sus métricas finales en estas dos corridas.
- No se alcanza la tolerancia numérica 1e-6 para esas dos redes, aun con seeds,
  flags y estados iniciales registrados. No se atribuye una causa concreta sin
  un experimento de diagnóstico; no se afirma reproducibilidad bit a bit.
- Se conservan ambas observaciones y se mantiene el protocolo de tiempos. El
  análisis separa equivalencia de datos/carga de repetibilidad de la trayectoria
  de aprendizaje. No se cambió el entorno para forzar coincidencia de resultados.
- Se añadió sellar_corridas_cpu.py para cerrar registros/manifiestos de corridas
  completas; comando: PCUDA\\Scripts\\python.exe -X utf8 scripts/sellar_corridas_cpu.py --experimento EXP-002.
- Auditoría adicional en el checkpoint CNN-A de la primera corrida: contador
  Adam 1440, coincidente con 96 lotes×15 épocas y sin sumar las 2 actualizaciones de
  warmup. La auditoría final verificará todos los checkpoints neuronales.


### Cierre de BIT-005 — Cinco corridas y resultados auditados

- Las cinco salidas de trabajadores fueron 0. La medición terminó el
  2026-11-17; el supervisor cerró la serie a 19:59:24.152376 UTC.
- Se ejecutaron 25 entrenamientos neuronales,375 épocas y36,000 actualizaciones
  Adam. Se comprobaron los 25 checkpoints finales: 1440 actualizaciones cada uno.
- Tensores, estados iniciales, fotos y paquetes coinciden entre las cinco
  corridas. Se verificaron los binarios iniciales, no solo sus metadatos, y la
  integridad de archivos sellados. No hay procesos benchmark activos al cerrar.
- Hallazgo en corrida 4: CNN-B obtuvo 0.928736 y CNN-A 0.930651; el programa seleccionó
  CNN-A+aug. Las otras cuatro seleccionaron CNN-B+aug. Eso cambia la carga total.
- DEC-015 se adoptó antes del resultado de la quinta corrida. Se conservaron las
  cinco observaciones y se estratificaron fases/ramas; no se excluyó la rápida.
- El supervisor cerró el controlador tras cinco salidas correctas. Se conserva
  serie-antes-cierre.json. La extensión06 únicamente preparó invocación/contexto;
  no produjo stdout, recursos ni tiempos de trabajador. Su registro explica por
  qué no es una sexta observación. No se alteró el benchmark congelado.
- Referencia común: suma de 14 intervalos medidos, no solapados, con las mismas
  cuatro redes base/PCA/dos bosques. Media 1119.9275623s; DE muestral 2.7231840s;
  CV 0.243157%; IC t95% nominal [1116.546283,1123.308842]s. Semiancho relativo 0.301919%.
- Total continuo adaptativo: media 1858.61650026s; DE 314.366s; CV 16.91%; rango
  1296.610–2011.095s. Es descriptivo de ramas distintas; no un baseline fijo GPU.
- La regla original de estrechar el IC global se suspendió por heterogeneidad,
  no se declaró cumplida. Se aplicó la referencia común descrita en DEC-015;
  la revisión de alcance es exploratoria y está revelada en el informe.
- Repetibilidad del aprendizaje no alcanzada en todas las redes con tolerancia
  1e-6; causa específica sin aislar. No se modificó PCUDA ni se instalaron paquetes.
- Postproceso ejecutado desde la raíz, después de terminar mediciones:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/sellar_corridas_cpu.py --experimento EXP-002
.\PCUDA\Scripts\python.exe -X utf8 scripts/preservar_datos_cpu.py --experimento EXP-002
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cpu.py --experimento EXP-002
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
```

- El supervisor de cierre se ejecutó en paralelo al controlador mediante
  PCUDA/scripts/cerrar_serie_cpu.py --experimento EXP-002; se registra el comando
  correcto con separadores de Windows en la [guía](reproducir-exp-002.md).
- Los scripts de sellado, archivado y análisis terminaron con código 0. La
  verificación documental final se registra en
  [verificacion-preparacion.md](../../resultados/preparacion/verificacion-preparacion.md).
- Copia local de 17,400 archivos:224,125,103 bytes, ZIP verificado y SHA-256
  `6af5baba0a6049b489263f5b6e159375f1e281ff4c23b61a950d91f557e1c23d`;
  fuente preservada sin descargar y fuera del intervalo medido.
- Cuatro gráficas renderizadas e inspeccionadas: referencia común, total
  adaptativo por rama, fases estratificadas y recursos CPU/RAM. Se ajustó la
  leyenda de recursos para no tapar el pico observado.
- [Informe definitivo](../../resultados/secuencial/EXP-002/resultados.md),
  datos crudos por RUN, CSV de estadísticas y scripts de regeneración disponibles.
  La guía de reproducción y su preparador no crean otra medición automáticamente;
  su entrada CLI se comprobó con --help. No se probó una máquina limpia.
- Paso2 completado con límites explícitos. Paso3, GPU, transferencias, speedup e
  informe PDF final siguen pendientes. Commit/revisión humana: pendientes.

## BIT-006 — Paso 3: oportunidades de paralelización

- Fecha UTC: 2026-11-17; consulta de identidades/cálculos a las 21:57:28 UTC.
  Responsable: Codex por solicitud del usuario; revisión humana pendiente.
- REQ-04/12; PROC-01/02/11; DEC-003/016; CAM-004. No genera EXP/RUN.
- Directorio de trabajo: raíz del repositorio. Se revisaron README de trazabilidad,
  decisiones abiertas, bitácora, requisitos, cambios y procedimientos antes de
  modificar. Se contrastó el texto normativo del paso 3 con el PDF, página 2.
- Entradas: notebook/PDF con sus hashes del inventario inicial; análisis estático
  del paso 1 y CSV de medias EXP-002. Identidades exactas en el documento del paso 3.
- Procedimiento: revisión del flujo y de las repeticiones; identificación de
  salidas independientes, reducciones y dependencias; priorización por tiempo CPU;
  selección de nueve familias y análisis de operaciones no seleccionadas.
- Lectura normativa realizada con Python del sistema, sin ejecutar el notebook:

```powershell
python -X utf8 -c "from pypdf import PdfReader; print(PdfReader('docs/Proyecto 3 CUDA.pdf').pages[1].extract_text())"
```

- Cálculos reproducibles: leer `media_s` de resumen.csv y dividir cada fase entre
  `media_s` de resumen_referencia_comun.csv, multiplicando por 100. Las cuatro
  fases entrenamiento_CNN-A/B y entrenamiento_FC-A/B suman 96.760341%; CNN-B
  representa 71.937664%. Se ejecutó esta derivación con csv de la biblioteca
  estándar en PCUDA; no se ejecutaron entrenamientos ni scripts de experimentos.
- Se comprobó SHA-256 con hashlib sobre PDF, notebook, los dos CSV y estrategia
  previa. El paso 3 conserva las identidades de entradas y CAM-004 la previa.
- Resultado: documento con selección principal CNN-B, operaciones complementarias,
  oportunidades secundarias y exclusiones. Se conserva preparación CPU según PDF.
  No se asignan hilos/bloques/grids ni se eligen kernels, librería o memoria.
- Artefacto: [oportunidades-paralelizacion.md](oportunidades-paralelizacion.md).
  Se actualizan índices, REQ-04, DEC-003/016 y PROC-11; no se alteran fuentes,
  configuraciones ni evidencia experimental sellada.
- Comandos de comprobación documental desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
git diff --check
```

- Resultado de comprobación: aprobado a las 22:00:06 UTC, salida 0: 206 enlaces
  válidos en 35 Markdown, 19 archivos originales con bytes/SHA-256 coincidentes,
  68 esquemas CSV consistentes y 125 filas experimentales enlazadas. `git diff
  --check` finalizó con salida 0; solo avisos de conversión LF/CRLF de Git.
  Evidencia en el [reporte de preparación](../../resultados/preparacion/verificacion-preparacion.md).
  Límite: verifica documentación e integridad, no compatibilidad CUDA, corrección
  numérica de una implementación GPU ni rendimiento. No se repiten mediciones CPU.
- Siguiente fase pendiente: paso 4, diseño del algoritmo paralelo; no iniciada.

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
