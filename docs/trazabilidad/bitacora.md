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

## BIT-007 — Paso4: diseño CUDA completo como propuesta

- Fecha:2026-11-17 UTC. Instantes de inspección registrados22:08:47 y22:17:26;
  cierre/validación indicado al finalizar. Responsable:Codex, solicitud del usuario.
- REQ-05/12; PROC-01/02/12; DEC-017/018/019/020; CAM-005. Sin EXP/RUN nuevos.
- Antes de modificar: lectura de AGENTS.md, README de trazabilidad, decisiones,
  bitácora, requisitos, cambios, procedimientos, análisis/mediciones previas y
  estrategia. Árbol limpio en commit2a14000a54fb38bc72dc9a45d5438ca726ae76bf.
- Inspección del notebook como JSON, celdas24/31–34; fuentes Keras locales Adam
  y BN; NPZ inicial CNN-B con allow_pickle=False. Lectura de página3 del PDF y
  contraste con apartados3–5 de página2. No se ejecutó el notebook ni entrenamiento.
- Se consultaron fuentes técnicas primarias NVIDIA/Numba/Keras/NumPy para
  APIs y modelo de ejecución. URLs y límites en la evidencia de inspección.
- Consulta nvidia-smi identifica GTX1650,4096MiB,sm_75,driver591.59. Metadatos
  PCUDA confirman que numba-cuda/numba/cuda-python no están instalados. No se
  reinstalan dependencias ni se descargan datos; lock compatible pendiente de
  admisión en paso5, sin certificar compatibilidad por el inventario.
- Cálculos derivados: arquitectura2245245parámetros entrenables y960BN móvil,
  arenas679.698MiB, gradientes parciales≤9MiB, payload por época y grids de ejemplo.
  NPZ80arreglos detecta cuatro estados RNG que deben preservarse como procedencia.
- Procedimiento de diseño: fijar contrato CNN-B; descomponer forward/backward;
  asignar salidas únicas, tiles16×16, particiones/reducciones; definir vida de
  buffers, transferencias y stream; fijar generación compartida de máscaras y
  secuencias; definir CPU pareada, validaciónCPU, pruebas/tolerancias y cronometraje.
- Errores conservados: una búsqueda de fuente BN con str.index sobre cadena
  inexistente devolvió ValueError; consulta corregida terminó0, sin cambios.
  Se rechazaron parches documentales con contextos inexistentes sin aplicar;
  se corrigieron por archivo y comprobó su contenido. Una búsqueda rg con glob
  de ruta Windows no expandido produjo error123; se corrigió usando filtro -g.
  No son fallos experimentales y no alteraron fuentes ni evidencia sellada.
- Artefactos: [diseño](diseno-paralelo.md), [kernels](diseno-kernels-cuda.md),
  [validación/medición](diseno-validacion-cuda.md) y
  [evidencia](../../resultados/diseno/paso-4-inspeccion.md), con comandos de
  inventario, resultados estáticos, identidades previas y fuentes consultadas.
- Decisiones: propuesta numba-cuda explícita, CPU para preparación/validación;
  nuevo EXP pareado necesario al implementar por cambio de control aleatorio y
  alcance. No usar el total histórico de14fases como baseline de una sola CNN.
- Petición complementaria del usuario: contrastar presentación Semana15 externa.
  Se leyó y copió por bytes con creación exclusiva a docs/referencias; hash
  acce03142813c9aca0bf4329000639bb2bb986f92e2810bd1c291774e73fab26.
  [Contraste](contraste-semana-15.md) con11temas de clase, equivalencias de API
  y precisiones. Se refina R(L,C) a32×8 para canales contiguos, sin cambiar
  presupuesto; su efecto de rendimiento queda sin medir. DEC-020.
- Comprobación aritmética estática: ty+8j con ty0..7,j0..255 cubre0..2047 una
  sola vez;16×16=32×8=256; shared2048/1024bytes según familia; parámetrosNPZ,
  1440pasos y95×128+20=12180 verificados. Copia de clase idéntica por bytes.
- Se actualizan REQ-05 y estado documental, manteniendo pendientes REQ-06/07/08/09
  en lo relativo a implementación, pruebas y resultados GPU.
- Comandos finales desde la raíz, PCUDA existente:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
git diff --check
git status --short
```

- Resultado final: aprobado a2026-11-17T22:29:58 UTC, salida0;252enlaces válidos
  en40Markdown,19originales con bytes/SHA-256 coincidentes,68esquemas CSV y125filas
  experimentales consistentes;56fuentes en captura temporal verificadas.
  git diff --check terminó0, solo avisos de conversión LF/CRLF.
  [Reporte mutable](../../resultados/preparacion/verificacion-preparacion.md).
  Límite: valida documentación, hashes, esquemas y sintaxis de scripts existentes;
  no compila kernels, ejecuta benchmarks ni prueba corrección/rendimiento CUDA.
- Siguiente paso:implementación paso5 cuando se solicite. No se inicia en esta
  actividad. Commit y revisión humana pendientes.

## BIT-008 — Implementación y ejecución CUDA, paso5

- Estado:completado para CNN-B;responsable:Codex,solicitud del usuario.
- REQ-06/07/09/12; CAM-006; DEC-021. Fuentes y evidencias previas revisadas antes
  de modificar; original intacto. Se aplica compute-env-setup para entorno/smoke.
- Inspección: PCUDA Python3.14.3,NumPy2.5.5,TF2.22.0rc0; sin Numba/CUDA Python.
  Toolkit local13.1; GTX1650,4096MiB,sm_75,driver591.59. Consulta nvidia-smi de
  inventario, sin entrenamiento. PDF páginas2/3 extraído con pypdf para contraste.
- Resolución preliminar, código0: PCUDA/python -m pip install --dry-run
  numba==0.67.0 llvmlite==0.49.0 numba-cuda cuda-python. No propone reemplazar
  NumPy existente; paquetes nuevos concretos se conservarán en reporte pip.
- Se registra antes de implementar la revisión que evita repetir CPU y sustituye
  la necesidad de mismas máscaras Dropout por pruebas puntuales de corrección.
- Actividad y evidencia se completarán con códigos de salida, RUN y resultados
  reales. Ningún resultado de rendimiento aceptado en este punto.

### Cierre con evidencia real

- Medición:2026-11-17T23:16:28–23:54:32UTC. EXP-003 congelado SHA-256
  00866b6fa0d48db134fdbb866b91cce2006c0ee7fa43812476337970a6cfb583.
  Cinco RUN completos,01..05,con códigos0;sin exclusiones de timing ni ampliación.
- Se aplicó DEC-021 por preferencia del usuario:sustituir la obligación de otra
  campaña CPU por reutilización de fase CNN-B compatible. No se entrenó CPU otra vez.
- Entorno:uv añadió15paquetes ausentes,75anteriores declarados sin cambios y
  pip25.3preexistente;NumPy/TF/Keras intactos. Caché uv fuera de sandbox requirió
  aprobación. Pipdescargas detenidas sin completar instalación;se preserva registro
  del intento. Detalles/comandos en [entorno](../../resultados/preparacion/cuda/entorno.md).
- Primer smoke detectó alias NumPy.row_stack retirado;alias local vstack sin
  modificar paquetes. CUDAreal pasó copia17valores. No simulator/fallbackCPU.
- Se implementaron kernels Python y NetCNN-B explícita;ensayo tiled0.6014s/lote
  y SGEMM0.2577s/lote justificaron DEC-022 antes de medir. Variante tiled conservada.
- Pruebas:operadores-00251checks y adjuntos-00178checks aprobados;b20/b128,
  FP64analítico/localTFCPU y tolerancias conservadas. Integracion-002aprueba10pasos,
  último20,1172muestras,counter,exportación,CPUeval sin mutación,save/load y restore.
- Fallos preservados:validacion-001/002/003 y rutas-004 no pasan identidad global
  estricta. Se verifican1y14rutas pooling distintas dentro de cota de redondeo
  para b2/b20;DEC-023 no las presenta como pruebas aprobadas. Introspección de
  recursos falló en integracion-001,se corrigió API y pasó en002 sin cambiar math.
- Notebookcopia83celdas,incluye68originales,códigoCPUguardadoFalse y apéndice de
  fuentes. Se ejecutó con PCUDA37celdas de código;ningún error de celda. Primer
  historialIPython fuera del workspace produjo avisosSQLite. Se ajustó ejecutor
  a historial en memoria y se guardó otra vista sin ejecutar la serie de nuevo.
  Ejecutor previo SHA-2566a6898edace43614aee9276d485d41dc276e464df0f9c0190657fecde77a6380,
  preservado en fuentes del primer RUN. Se conserva salida previa y avisoTCP local.
- [Salida notebook validado](../../resultados/paralelo/EXP-003/notebook-validado.salida.json):
  código nativo0,misma huella de serie antes/después,sin nuevas corridas. La shell
  PowerShell había marcado stderr nativo como error aunque las celdas completaron;
  el supervisor Python registra el código real. No se repitieron entrenamientos.
- Cada RUN verifica código/dataset y bytes X/y frente a CPU,restaura después
  del warmup,cuenta12180muestras/época y1440updates,exporta Adam real. Se auditan
  los cinco modelos HDF5:contador1440,76datasets,contenido canónico idéntico
  47f46467b40415af622ef099aef5a8d125e43c8ae62bd69b91dadb574ebe49f1.
- Tiempos entrenamiento GPU:415.499,418.797,420.509,419.772 y418.461s
  (redondeados;valores crudos en CSV);media418.607683s,DE1.916308s,CV0.4578%,ICt95%[416.228272,420.987094]s.
  Semiancho0.5684%≤5%:cinco bastan según regla previa. CPUmedia805.649726s,
  speedup1.924594×. Transferencias yCPUvalidación se incluyen,anidadas sin doble suma.
- Calidad:accuracy0.986206889,loss0.061565157,en todas las GPU;no atribuir mejora
  de calidad a hardware ni prometer idéntica trayectoria CPU. Referencia histórica
  no alternada,shuffle/máscaras diferentes y cambioCPUworkers1→6 explícitos.
- Cuatro gráficas renderizadas e inspeccionadas;PDF8páginas con tablas,gráficas
  y capturas. Capturas seleccionadasCPU23:39:47/GPU23:37:06 correlacionadas con
  recursos.csv. IntentoDPI previo incompleto/no utilizable conservado y excluido.
- [Informe](../../resultados/paralelo/EXP-003/resultados.md),fuentes de análisis,
  comparabilidad por criterio,CSV clasificado por ejecutor,checkpoints,logs y
  [reproducción](reproducir-cuda.md) disponibles. Modelos grandes fuera de Git.
- Comandos principales desde raíz (salidas y comando exacto por RUN en invocacion.json):

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cuda.py prepare --experimento EXP-003
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/ejecutar_notebook_cuda.py --salida resultados/paralelo/EXP-003/notebook-ejecutado.ipynb
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cuda.py --experimento EXP-003
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/sellar_resultados_cuda.py --experimento EXP-003
git diff --check
```

- Verificación documental aprobada2026-11-18T00:11:31UTC (17noviembre local):
  352enlaces en45Markdown,19originales con bytes/SHA-256 coincidentes,
  107esquemasCSV y214filas de índices experimentales,76fuentes de prueba archivadas.
  Salida0;git diff --check0,solo avisos LF/CRLF. No es otra ejecución de entrenamiento.
- Revisión posterior de la variante tiled conservada:operadores-explicit-003
  aprueba51checks locales;no se mezcla con la serie de rendimiento cuBLAS.
  Los archivos de evidencia quedan cerrados mediante sellar_resultados_cuda.py;
  originales yCPUEXP-002 permanecen intactos.
- Pendiente:revisión del equipo,contribuciones/commit y entrega global. No se
  certifica reconstrucción desde cero en otra máquina ni ocupación por SM.

## BIT-009 — Incrustar salidas reales en notebook

- Responsable:Codex,por solicitud explícita posterior del usuario;CAM-007.
- Origen:último RUN completo EXP-003,RUN-20261117T234658Z-gpu-05. No se crea EXP
  ni RUN:solo se visualizan datos experimentales existentes. UTC de cierre en
  resultados/notebook-salidas/validacion.json.
- Se conserva la fuente congelada53050fbcc27a8629850d36165c5c4387982714277f09a58e256e3537b051b86f
  y se crea otra copia con salidas al comienzo. Se añade --entrada al ejecutor,
  registrando su identidad previa en CAM-007;ningún módulo de entrenamiento cambia.
- Se ejecuta la plantilla con PCUDA,historial en memoria,GPUFalse y hardware
  recuperado de hardware.json. Se imprimen literalmente15líneas EPOCH de stdout,
  tabla15épocas,métricas,testloss,tiempos reales,curva última corrida y hashes.
- Código nativo0;96celdas totales,44ejecutadas,0errores y5imágenes incrustadas.
  La huella de serie antes/después es
  4870e1f1d039f88822eccabb0339ef246c9bfbb8351a4e44588c3cb6b1639935.
  Las184entradas del manifiesto EXP-003 mantienen SHA-256;ningún entrenamiento se repite.
- [Notebook entregado](../../notebooks/Lab3_ASL_CUDA_con_salidas.ipynb),
  [procedimiento](../../resultados/notebook-salidas/README.md),
  [salida nativa](../../resultados/notebook-salidas/salida.json) y
  [validación](../../resultados/notebook-salidas/validacion.json).
- Comandos desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/notebook_con_salidas.py
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/ejecutar_notebook_cuda.py --entrada resultados/notebook-salidas/plantilla.ipynb --salida notebooks/Lab3_ASL_CUDA_con_salidas.ipynb
```

- El segundo se lanzó con subprocess.run y stdout/stderr a archivos para
  registrar su código nativo sin confundir avisos de PowerShell con un fallo.
  Comando exacto en salida.json. Se actualizan README e índice de trazabilidad.
- Validación documental final aprobada el 2026-11-18 a las 02:42:52 UTC con
  `.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py`:
  364 enlaces locales válidos en 46 Markdown, 108 esquemas CSV consistentes,
  19 identidades originales conservadas y 78 fuentes archivadas verificadas.
  `git diff --check` sin errores. Esta validación no ejecutó entrenamiento.

## BIT-010 — Verificación de pasos 1–5 antes de conclusiones

- Fecha:2026-11-18 UTC,17noviembre local. Responsable:Codex por solicitud
  explícita del usuario;CAM-008;REQ-02 a REQ-12. Sin nueva campaña experimental.
- Se leyó directamente el PDF de3páginas,documentación,decisiones y código CPU/
  CUDA. La extracción inicial falló en codificación cp1252;se usó python -X utf8
  con pymupdf ya instalado en Python del sistema. No se alteró ni descargó el PDF.
- Se implementó auditor estático sin importar modelos. Primer intento conservado
  en resultados/auditoria/pasos-1-5/verificacion.json:7controles correctos y
  KeyError por encabezado ruta del manifiesto GPU. Se adaptó el lector y se
  exigió --destino para conservar intentos separados.
- Segundo intento:02:50:06–02:50:30UTC,código0,80controles aprobados. Se leyeron
  17,400imágenes solo para SHA-256;10manifiestos RUN,184entradas globales y fuentes
  congeladas intactas;hashes de estados/modelos y registros de tensores coinciden.
  Medias/DE recalculadas desde crudos y cocientes publicados consistentes.
- Comandos desde raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/auditar_pasos_1_5.py
.\PCUDA\Scripts\python.exe -X utf8 scripts/auditar_pasos_1_5.py --destino resultados/auditoria/pasos-1-5-revision-02
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
git diff --check
```

- El primero corresponde a la versión inicial anterior a exigir --destino;
  esa invocación conserva su error y SHA del script en su JSON. No se repite.
- Revisión visual de capturas CPU/GPU:legibles y con muestra/UTC/RUN correlacionados.
  CPUprep225.2% y GPUentrenamiento88%,1213/4096MiB:demuestran carga,no saturación
  total ni ocupación porSM. La reserva REQ-09 queda explícita como parcial.
- Se verificaron salidas notebook:15líneas idénticas a stdout real y0errores.
  51operadores/78adjuntos aprobados son antecedentes,no pruebas nuevas;diagnósticos
  globales estrictos permanecen fallidos. No se certifica equivalencia bit a bit.
- [Dictamen y detalle](revision-pasos-1-5.md):se puede avanzar al paso6 para CNN-B
  con límites históricos/RNG/workers,calidad y saturación. Sin inconsistencia
  detectada que exija repetir entrenamientos. No se inicia todavía el paso6.
- Evidencias:[auditoría aprobada](../../resultados/auditoria/pasos-1-5-revision-02/verificacion.json)
  y [primer intento conservado](../../resultados/auditoria/pasos-1-5/verificacion.json).
- Validación documental posterior:resultado en
  [verificación de preparación](../../resultados/preparacion/verificacion-preparacion.md).
  No implica ejecución del entrenamiento. Equipo y entrega global pendientes.

## BIT-011 — Paso 6: análisis y conclusiones CNN-B

- Fecha:2026-11-18UTC,17noviembre local. Responsable:Codex por solicitud del usuario;
  DEC-024;CAM-009;PROC-09;REQ-08/10/12. Sin nuevo EXP/RUN ni entrenamiento.
- Entradas:5CPU EXP-002,5GPU EXP-003,configuraciones,CSV por RUN,calidad,recursos,
  PDFnormativo y auditoría previa.27fuentes con bytes/SHA-256 en fuentes.csv.
- Se contrastó apartado6 directamente con página2 del PDF usando pymupdf existente
  en Python del sistema. Se revisaron README/decisiones/bitácora y reservas previas.
- Se aplicó figure-style:observaciones individuales,unidades,DE declarada,colores
  consistentes y revisión visual. Script usa PCUDA y Matplotlib ya instalado;
  no requiere descarga,entrenamiento,CUDA ni exportación PDF.
- Se verificaron sello184entradas y5manifiestosCPU,leyeron10tiemposporRUN,
  recalcularon estadísticas y cocientes de medias,contrastándolos con CSVprevio.
  Sin emparejar RUNhistóricos ni excluir observaciones por su tiempo.
- Primera generación03:11:49–03:11:52UTC completó con código0. Revisión visual
  detectó nota redundante cortada en figura2. Se preservó fuente inicial y carpeta
  completa;se ajustaron bbox/padding,redacción de residuo y condiciones. Se ejecutó
  nuevamente solo posprocesado en destino nuevo;ningún dato/cifra cambió.
- Comandos desde raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/analizar_paso_6.py --destino resultados/comparacion/paso-6
.\PCUDA\Scripts\python.exe -X utf8 scripts/analizar_paso_6.py --destino resultados/comparacion/paso-6-revision-02
.\PCUDA\Scripts\python.exe -X utf8 scripts/verificar_preparacion.py
git diff --check
```

- El primer comando corresponde a fuente inicial preservada;el segundo,a fuente
  vigente. Comando exacto,intérprete,UTC y versiones en cada procedencia.json.
  No regenerar sobre carpeta existente;usar nueva identidad de salida.
- [Análisis vigente](../../resultados/comparacion/paso-6-revision-02/analisis-y-conclusiones.md):
  tablas10RUN,4fases,dispersión,desglose,calidad,9factores,5conclusiones y3gráficas
  PNG/SVG. Resultado principal805.6497257→418.607683s,387.0420427s ahorrados,
  speedup1.924593739×,reducción48.040982%. Transferencias3.527822618s incluidas.
- Revisión visual final:tres imágenes legibles,ejes/leyendas/anotaciones completos.
  Se verificaron por Python/PIL hashes de13derivados/27fuentes,sumaporcentajes100,
  tres PNGválidos,ausenciaPDFnuevo e identidad de fuente preliminar conservada.
  [Cierre](../../resultados/preparacion/paso-6-borrador/validacion-cierre.json)
  aprobado03:14:17UTC;no entrenamiento.
- Las causas se clasifican por evidencia;sin perfilado causal exclusivo por kernel,
  sin afirmar corrección global absoluta ni saturaciónSM/VRAM. PreparaciónCPU1→6
  separada del beneficioCUDA;testCPU no mejora. Reserva histórica conserva alcance.
- Índices/requisitos actualizados;verificación documental en informe de preparación.
  Siguiente trabajo autorizado aquí termina en paso6Markdown;informefinal/PDF/equipo
  y entrega pendientes de otra actividad. No se editan fuentes congeladas ni RUN.

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
