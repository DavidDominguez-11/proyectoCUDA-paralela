<!-- Slide number: 1 -->
De CPU a CUDA
Reconocimiento ASL · del archivo a la predicción



























































































3 redes entrenadas en GPU
1.92×
CNN-B
CPU / CUDA · media de 5 pruebas
3.35×
FC-A
CPU / CUDA · media de 5 pruebas
4.86×
FC-B
CPU / CUDA · media de 5 pruebas
David Domínguez 23712  ·  Jonathan Diaz 23873  ·  Luis Padilla 23663
1
NOTAS: Presentar el reconocimiento de señas ASL como caso de aplicación. El proyecto parte del laboratorio original, mide una referencia CPU y traslada el entrenamiento completo de CNN-B, FC-A y FC-B a CUDA Python. Los tres números son speedup de entrenamiento total medido, con transferencias y validación CPU incluidas. No son aceleración de todo el notebook.

Procedencia / respaldo: Comparación ampliación FC, n=5 por modelo; guía Proyecto 3 CUDA
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 2 -->
El proyecto en seis pasos

01
Analizar
Flujo y operaciones

02
Medir CPU
5 corridas equivalentes

03
Seleccionar
Matemática repetitiva

04
Diseñar
Threads · blocks · grids

05
Implementar
CUDA Python y cuBLAS

06
Comparar
Tiempo · speedup · calidad
Misma carga de trabajo → mediciones conservadas → resultados verificables
2
NOTAS: Seguir el mismo orden que la guía. Los diagramas resumen el procedimiento; después se concentra la exposición en tiempos, variabilidad, calidad y evidencia de recursos. La extensión FC está integrada al diseño y a las conclusiones finales.

Procedencia / respaldo: Guía principal, apartados 1–6
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 3 -->
De 87,000 imágenes a una prueba controlada
A
B
C
M
N
S
29
clases
600 imágenes por clase
17,400
imágenes seleccionadas
RGB · 64×64 · float32

/255 → [0,1]
Mismos tensores CPU / GPU
GRÁFICA NATIVA: BAR_STACKED (58)
Train: [70.0]
Validación: [15.0]
Test: [15.0]
Train 12,180   ·   Validación 2,610   ·   Test 2,610
3
NOTAS: El dataset ASL Alphabet contiene 29 clases, letras A–Z y las clases del, nothing y space. Se seleccionaron 600 imágenes por clase, 17,400 en total, con particiones estratificadas 70/15/15. Las seis fotografías provienen de archivos reales del manifiesto de entrenamiento; no son imágenes generadas. Se trabajó con archivos almacenados, no con video en vivo.

Procedencia / respaldo: Manifiesto EXP-002; selección y hashes de seis archivos en datos-presentacion.json
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 4 -->
1 · ¿Qué hacía el programa original?

01
Archivos JPEG
Dataset local


02
Preparación
RGB → 64×64 → /255


03
Particiones
Train / val / test


CNN-A / CNN-B
Convoluciones

FC-A / FC-B
Capas densas

PCA + RF
Modelos clásicos


Evaluar → comparar → aumento de imágenes → guardar → predecir
La versión original también usa paralelismo interno de librerías CPU; “secuencial” identifica la referencia
4
NOTAS: Describir la entrada de imágenes, la selección y transformación a tensores. El laboratorio entrena cuatro redes base y una variante con aumento, además de PCA y Random Forest. Después compara, guarda e infiere sobre fotos del equipo. La rama aumentada cambia según la selección: por eso su total adaptativo no se usó como denominador de speedup. El original se conserva intacto.

Procedencia / respaldo: Notebook original y análisis original; CPU EXP-002
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 5 -->
1 · Las operaciones que se repiten
Operación identificada | Datos que procesa | Repeticiones por modelo
Abrir / transformar archivos | 17,400 imágenes | 1 por imagen preparada
Recorrer datos de entrenamiento | 12,180 imágenes | 15 épocas
Formar lotes | Batch 128; último 20 | 96 lotes / época
Forward + pérdida + backward | Activaciones y pesos | 1,440 lotes
Actualizar Adam | Pesos + momentos + velocidades | 1,440 actualizaciones
Validar en CPU | 2,610 imágenes | 15 pasadas
Evaluar al final | 2,610 imágenes test | 1 pasada

182,700 imágenes vistas durante entrenamiento
5
NOTAS: Explicar cómo una imagen termina generando miles de operaciones por capa. Se cargan 17,400 imágenes; cada red procesa 12,180 por época, en 96 lotes, incluyendo uno de 20. Quince épocas son 1,440 actualizaciones Adam y 182,700 presentaciones de imágenes de entrenamiento por modelo. Las 15 validaciones procesan 39,150 imágenes y no actualizan pesos. Estos son los volúmenes de la campaña fija, no el EarlyStopping histórico del notebook original.

Procedencia / respaldo: Análisis original; configuraciones EXP-002/003/012
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 6 -->
2 · Referencia CPU: medir antes de cambiar

01
Verificar
Datos + estado inicial

02
Calentar
Restaurar después

03
Cronometrar
perf_counter_ns
Corrida | 14 fases comunes (s)
CPU 1 | 1119.751
CPU 2 | 1118.294
CPU 3 | 1122.869
CPU 4 | 1122.335
CPU 5 | 1116.389
1119.93 s
media de la referencia común
DE 2.72 s  ·  CV 0.24%  ·  n=5

15 épocas fijas · batch 128 · FP32
El total adaptativo (1,858.62 s) se conserva aparte: no se utiliza para calcular speedup
6
NOTAS: La referencia no es estrictamente monohilo. TensorFlow CPU usa seis hilos intra-op y uno inter-op, BLAS seis; Random Forest mantiene n_jobs=-1. El calentamiento y las importaciones quedan fuera; se restaura el estado antes de medir. Las cinco corridas completas válidas son CPU01–05. La carga base fija comparte 14 fases y tiene variación baja. El total adaptativo mezcla distintas CNN aumentadas y no es una referencia homogénea para CUDA.

Procedencia / respaldo: EXP-002: serie.json, resultados.md y resumen_referencia_comun.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 7 -->
2 · CNN-B concentraba el tiempo CPU
GRÁFICA NATIVA: DOUGHNUT (-4120)
Tiempo CPU: [805.6497256999999, 124.83926084, 102.44108204, 50.715659880000004, 36.28183383999999]
71.94%
CNN-B
Entrenamiento | Tiempo CPU (s) | Parte del total
CNN-B | 805.65 | 71.94%
FC-B | 102.44 | 9.15%
FC-A | 50.72 | 4.53%
CNN-A | 124.84 | 11.15%
Prioridad inicial: CNN-B
Ampliación final: FC-A y FC-B
7
NOTAS: Se suman las medias de 14 fases no solapadas. CNN-B representa 71.94% del total común. FC-A y FC-B aportan más tiempo de entrenamiento paralelizable; con las tres se cubre aproximadamente 85.61% de esta referencia. El donut no representa el pipeline adaptativo completo. La validación dentro de fit ya está incluida y no se añade otra vez.

Procedencia / respaldo: CPU EXP-002 resumen.csv; referencia común 1119.9275623 s
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 8 -->
3 · ¿Qué se paralelizó y por qué?
Operación | ¿Qué puede ocurrir a la vez? | Ejecutor final
Archivos / decodificación / resize | Imágenes independientes | CPU · 6 workers
Convoluciones / Dense | Productos y sumas de matrices | GPU · cuBLAS
ReLU / Dropout / normalización | Elementos y características del lote | GPU · kernels Python
Pérdida / gradientes / Adam | Muestras, parámetros y reducciones | GPU · kernels Python
Validación / test | Predicciones del modelo exportado | CPU
PCA / RF / CNN-A / augmentation | Fuera del alcance CUDA elegido | CPU original

CNN-B + FC-A + FC-B: entrenamiento completo en GPU
8
NOTAS: No se trasladó todo el programa al GPU. Se seleccionaron la matemática del entrenamiento de tres redes y sus operaciones por elemento o por reducción. La carga de archivos queda en CPU porque implica rutas, sistema de archivos, decodificación e I/O irregular. PCA, Random Forest, CNN-A y augmentation no se portaron en este alcance. La validación y test siguen en CPU por el diseño aplicado; sus costes se declaran.

Procedencia / respaldo: Paso3, implementación revisada y ampliación FC
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 9 -->
4 · Un flujo CPU ↔ GPU explícito

CPU / HOST
Archivos y decodificación
RGB / resize / normalización
6 workers
Lotes y máscaras Dropout

GPU / DEVICE
Forward + pérdida
Backward / gradientes
Adam + métricas





















CPU / EVALUAR
Validación cada época
Test final
Métricas / guardado

H2D

D2H
Por época: 96 lotes → exportar → validar en CPU → continuar hasta 15 épocas
9
NOTAS: Recorrer la separación de fases. CPU realiza lectura y preparación, genera el orden de lotes y máscaras y usa memoria host fijada para copiar. H2D envía datos al stream. GPU hace forward, pérdida, backward y Adam. Al final de cada época, D2H exporta pesos y estadísticas BN para validar en CPU. El ciclo se repite quince veces. Preparación no se presenta como CUDA y test no se presenta como inferencia GPU.

Procedencia / respaldo: Diseño aplicado: modelo Network/FCNetwork, un stream, sin solapamiento
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 10 -->
4 · Tres redes, la misma regla de reparto

CNN-B

64×64×3

Conv 32 → Conv 32

Pool → Conv 64 → Conv 64

Pool → Conv 128 → Pool

Flatten → Dense 256

Softmax 29
2,245,245
parámetros entrenables
5 Conv · 3 Pool · BN momentum 0.9

FC-A

64×64×3

Flatten 12,288

BatchNormalization

Dense 128 → Dense 64

Dropout 0.3 por capa oculta

Softmax 29
1,607,709
parámetros entrenables
BN de entrada · momentum 0.99

FC-B

64×64×3

Flatten 12,288

BatchNormalization

Dense 512 → 256 → 128

Dropout 0.4 por capa oculta

Softmax 29
6,484,509
parámetros entrenables
BN de entrada · momentum 0.99
10
NOTAS: FC significa fully connected: red de capas densas, diferente de la CNN que aprende filtros espaciales. CNN-B contiene cinco convoluciones con 32/32/64/64/128 filtros, tres pooling y Dense256. Las FC aplanan los 12,288 valores de la imagen y usan BN al inicio. Se preservan las arquitecturas, incluyendo Dropout y la diferencia de momentum BN. Las cifras son parámetros entrenables, no el número de operaciones ni la VRAM observada.

Procedencia / respaldo: Construcciones originales + FCNetwork; inicializaciones históricas preservadas
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 11 -->
4 · Threads, blocks y grids
































THREADS
elementos independientes






BLOCKS
hilos agrupados















GRID
todos los bloques del kernel
Operación | Block | Grid / asignación
ReLU / Dropout / Adam | 256 hilos | ceil(E / 256), tope 4,096
BN parcial FC, C=12,288 | 32 × 8 hilos | 1 × 384 bloques
BN final FC | 256 hilos | 48 bloques
Softmax / pérdida | 32 hilos | 128 bloques; último lote: 20
Matrices Dense / Conv | cuBLAS SGEMM | Distribución interna de NVIDIA
11
NOTAS: Explicar los tres niveles CUDA con el dibujo. Los kernels propios distribuyen elementos en hilos de bloques de 256 y usan recorrido grid-stride. La reducción parcial BN usa 32×8; la final FC asigna un hilo por característica. Softmax usa 32 hilos y un bloque por muestra. Los productos SGEMM se delegan a cuBLAS: su reparto interno es de NVIDIA y no se atribuye al equipo.

Procedencia / respaldo: Código CUDA medido; recursos compilados y lanzamientos por RUN
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 12 -->
5 · Qué se implementó y qué se midió

Numba-CUDA
Kernels Python explícitos

cuBLAS SGEMM
Matrices FP32 · modo PEDANTIC

TensorFlow / Keras
Solo CPU: validar y guardar

Fuera del entrenamiento medido
Warmup → restaurar → preparar imágenes

Staging / máscaras
H2D
Forward / backward / Adam
D2H
Validación CPU
← Tiempo pared sincronizado: todo el bloque está incluido →
Después: test CPU cronometrado aparte; guardado y gráficas fuera de fit
12
NOTAS: La implementación usa Numba-CUDA para kernels Python y cuBLAS SGEMM para los productos grandes; TensorFlow/Keras está restringido a CPU y espejo para validar/guardar. El intervalo principal es pared sincronizada y contiene staging/máscaras, H2D, forward/backward/Adam, D2H y validación CPU. Preparación de imágenes, warmup/compilación y guardado/figuras se separan. Un único stream mantiene el orden y no hay solapamiento.

Procedencia / respaldo: EXP-003 y colección FC EXP-011/012; fuentes congeladas y temporizadores reales
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 13 -->
5 · Verificar antes de confiar en el tiempo
51
pruebas de operadores CNN-B
Tolerancias locales verificadas
78
adjuntos / gradientes CNN-B
Activaciones y dY compartidos
4
casos FC-A / FC-B
Lotes 20 y 128 · error probabilidades ≤1.67×10⁻⁶
1,440
pasos Adam por corrida
Checkpoint real · valores finitos
Pruebas numéricas locales aprobadas ≠ trayectoria completa CPU/GPU idéntica
13
NOTAS: Las pruebas locales contrastan operadores y gradientes con un oráculo CPU. Se verificó la restauración tras warmup y el contador Adam real de los checkpoints. La corrección local aprobada no implica una trayectoria numérica idéntica durante quince épocas: RNG, orden y redondeo difieren y en la CNN hubo diagnósticos estrictos de rutas cerca de empates que no se etiquetaron como aprobados. Se conserva esa reserva.

Procedencia / respaldo: Pruebas operadores/adjuntos CNN-B; validacion-01 FC; auditoría HDF5 final
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 14 -->
5 · CPU real durante su fase de trabajo
507.3%
CPU proceso + hijos
100% equivale a un núcleo
0%
GPU en esta muestra
Captura durante CPU
Monitor en vivo · lectura original psutil / nvidia-smi · muestra sostenida, época 3
14
NOTAS: Esta captura pertenece a la ejecución adicional n=1 del notebook CPU, separada de los promedios de la campaña. Fue tomada durante entrenamiento CNN-B en CPU, no durante preparación. psutil usa 100% por procesador lógico, por eso 507.3% indica varios núcleos activos. GPU estaba a0%. La evidencia de preparación con seis workers se presenta en el respaldo de recursos; no atribuir esta captura a I/O.

Procedencia / respaldo: EXP-008 captura-cpu-sostenido.png y JSON de muestra original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 15 -->
5 · GPU real durante entrenamiento CNN-B
91%
actividad GPU
Agregado nvidia-smi
1,213
MiB de VRAM
29.6% de los 4,096 MiB
Actividad GPU ≠ ocupación por SM; VRAM utilizada ≠ saturación de toda la memoria
15
NOTAS: La captura es de la corrida adicional de notebook CUDA y no modifica las cinco medias previas. nvidia-smi registró91% de actividad GPU y1,213MiB de4,096, un29.6% de capacidad VRAM. Actividad agregada no significa91% de núcleos ni ocupación por SM. Es evidencia de uso real y actividad elevada; no prueba saturación total de memoria.

Procedencia / respaldo: EXP-009 captura-cuda.png y metadata original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 16 -->
5 · FC-A y FC-B también usaron CUDA
FC-A  ·  50% GPU  ·  111 MiB
FC-B  ·  32% GPU  ·  187 MiB
Capturas puntuales reales; los promedios y máximos del monitor están en el anexo
16
NOTAS: Son capturas originales durante la primera corrida válida de cada FC. FC-A registró50% de actividad y111MiB; FC-B32% y187MiB en esas muestras. Las capturas no son máximos globales. Una GPU menos activa puede terminar antes cuando tiene menos trabajo por lote o mayores proporciones de host y validación; el speedup se determina por tiempo total. No se llenó VRAM artificialmente.

Procedencia / respaldo: EXP-011 FC-A1 y EXP-012 FC-B1: capturas y CSV originales
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 17 -->
6 · Tres redes, menos tiempo total
CNN-B
GRÁFICA NATIVA: COLUMN_CLUSTERED (51)
Tiempo: [805.6497256999999, 418.607683]
FC-A
GRÁFICA NATIVA: COLUMN_CLUSTERED (51)
Tiempo: [50.715659880000004, 15.122541380000001]
FC-B
GRÁFICA NATIVA: COLUMN_CLUSTERED (51)
Tiempo: [102.44108204, 21.06199454]
Entrenamiento | CPU media ± DE (s) | CUDA media ± DE (s) | Speedup | Menos tiempo
CNN-B | 805.65 ± 2.72 | 418.61 ± 1.92 | 1.92× | 48.04%
FC-A | 50.72 ± 1.26 | 15.12 ± 0.33 | 3.35× | 70.18%
FC-B | 102.44 ± 3.28 | 21.06 ± 0.20 | 4.86× | 79.44%
Escalas distintas por panel · Speedup = media CPU / media CUDA · n=5 · DE muestral
17
NOTAS: Esta es la tabla principal de resultados. Cada cifra resume cinco corridas por dispositivo/modelo. CUDA incluye transferencias y validación CPU. Los tres minigráficos tienen escalas verticales diferentes: comparan CPU/GPU dentro de cada arquitectura. Mostrar la fórmula speedup como cociente entre medias, no media de cocientes y no suma de speedups. La CPU es histórica; no se repitió la campaña para preparar la ampliación.

Procedencia / respaldo: comparacion/ampliacion-fc/comparacion.csv y estadisticas.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 18 -->
6 · Las cinco repeticiones, sin escoger la mejor
RUN | CNN-B CPU | CNN-B CUDA | FC-A CPU | FC-A CUDA | FC-B CPU | FC-B CUDA
1 | 803.847 | 415.499 | 48.520 | 14.829 | 100.825 | 21.303
2 | 806.870 | 418.797 | 51.495 | 15.686 | 100.347 | 20.945
3 | 802.320 | 420.509 | 51.378 | 14.970 | 108.090 | 21.251
4 | 809.363 | 419.772 | 50.742 | 15.000 | 102.501 | 20.959
5 | 805.848 | 418.461 | 51.443 | 15.128 | 100.441 | 20.852
0.46%
CNN-B · CV CUDA
2.20%
FC-A · CV CUDA
0.96%
FC-B · CV CUDA
Tiempos en segundos · 15 épocas por red y corrida · sin mezclar las observaciones adicionales n=1
18
NOTAS: Mostrar todas las observaciones de entrenamiento. No se eligió solo la ejecución más rápida. FC-A1 guardada en EXP-011 se reutilizó después de corregir solo la presentación; EXP-012 hizo las nueve pendientes. Un intento FC-A incompleto de EXP-010 está excluido del promedio por no cerrar/checkpointear, no por su velocidad. Las cinco repeticiones CUDA parten del mismo estado y RNG: la variación resume tiempo, no cinco aprendizajes independientes.

Procedencia / respaldo: observaciones.csv; series CPU01–05, GPU01–05, FC colección EXP-012
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 19 -->
6 · Comparar fases sin mezclar causas
Fase / alcance | Referencia (s) | Variante (s) | Cociente | Qué lo explica
Preparación CPU | 17.325 | 8.034 | 2.156× | CPU: 1 → 6 workers
Entrenamiento CNN-B | 805.650 | 418.608 | 1.925× | CUDA + transferencias + val.
Evaluación final CPU | 2.322 | 2.492 | 0.932× | Ambas en CPU
Suma de tres fases | 825.297 | 429.134 | 1.923× | Suma derivada; solo CNN-B
958.81 s
suma de tres entrenamientos CPU
Comparación derivada del mismo alcance
454.79 s
suma con entrenamiento CUDA
Cociente agregado: 2.108×
19
NOTAS: La preparación mejoró de17.325s a8.034s con seis workers CPU, cociente2.156x: no es aceleración CUDA. El test final continúa CPU y no mejoró. La suma comparable de preparación, entrenamiento CNN-B y test pasa825.297 a429.134s:1.923x, pero no es el pipeline completo. Como cálculo aparte, sumar los tres entrenamientos da958.806 a454.792s y2.108x, sin medir un pipeline continuo ni añadir los cocientes individuales.

Procedencia / respaldo: paso-6-revision-02/comparacion_fases.csv y tres medias de fit
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 20 -->
6 · ¿Dónde se va el tiempo con CUDA?
GRÁFICA NATIVA: BAR_STACKED (58)
Matemática / stream: [81.41629831554482, 36.692817917515434, 44.423654430937205]
H2D: [0.7914144817128956, 11.460734155029435, 8.849404169529441]
D2H: [0.051337039955532245, 1.227734127583626, 1.0570486991282089]
Validación CPU: [7.721692910268023, 23.021126229775277, 23.779007588854107]
Resto host: [10.019257252518722, 27.597587570096227, 21.890885111551043]
Modelo | Stream (s) | H2D (s) | D2H (s) | Val. CPU (s) | Host (s)
CNN-B | 340.815 | 3.313 | 0.215 | 32.324 | 41.941
FC-A | 5.549 | 1.733 | 0.186 | 3.481 | 4.173
FC-B | 9.357 | 1.864 | 0.223 | 5.008 | 4.611
Componentes anidados: NO se suman otra vez al tiempo total · Stream ≠ cómputo activo exclusivo
20
NOTAS: El gráfico normaliza cada entrenamiento a100% para comparar proporciones; los tiempos absolutos de las tres redes son muy diferentes. CNN-B pasa la mayor parte del intervalo en matemática del stream. En las FC crecen las proporciones de validación CPU y administración. H2D/D2H son costes incluidos. Los eventos del stream pueden contener huecos de envío del host; no significan tiempo activo exclusivo de un kernel. La tabla muestra segundos para evitar confundir porcentaje con tiempo absoluto.

Procedencia / respaldo: Desglose CNN-B paso6 y ampliación FC/desglose.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 21 -->
6 · CNN-B: aprendizaje durante 15 épocas
GRÁFICA NATIVA: LINE_MARKERS (65)
CPU: [20.643678307533264, 52.57471263408661, 64.55172419548035, 80.22988438606262, 78.86589884757996, 86.81226015090942, 89.1034471988678, 94.83524918556213, 92.83524870872498, 93.64750862121582, 93.70114922523499, 97.33333349227905, 95.08812308311462, 95.63218593597412, 96.91954135894775]
CUDA: [12.605364620685577, 37.04980909824371, 57.85440802574158, 77.8927206993103, 92.1072781085968, 85.28735637664795, 88.81226181983948, 91.99233651161194, 83.0651342868805, 94.90421414375305, 91.99233651161194, 91.9157087802887, 97.62452244758606, 96.20689749717712, 98.58237504959106]
GRÁFICA NATIVA: LINE_MARKERS (65)
CPU: [11.57828197479248, 2.956112790107727, 1.84871826171875, 0.9789945960044861, 0.9488107204437256, 0.6191243797540664, 0.3982876241207123, 0.20408490896224976, 0.24831838756799698, 0.23587343692779542, 0.19909692406654358, 0.09991313219070434, 0.15459380447864532, 0.14880216047167777, 0.10005851201713085]
CUDA: [21.91301155090332, 8.168072700500488, 2.5862271785736084, 1.0154019594192505, 0.31078094244003296, 0.5141513347625732, 0.399549663066864, 0.37006038427352905, 0.6767783761024475, 0.19163855910301208, 0.26367589831352234, 0.24109329283237457, 0.09185357391834259, 0.13204194605350494, 0.04838373884558678]
Media de 5 corridas por implementación · pesos iniciales iguales · RNG interno distinto
21
NOTAS: Los trazos son medias de cinco trayectorias por implementación y se calculan a partir de historiales guardados. AzulCPU, naranjaCUDA. La validación se ejecuta en CPU en ambas variantes, con los pesos exportados cuando el entrenamiento es CUDA. Los resultados no son una ejecución adicional. Curvas parecidas no prueban identidad de gradientes ni de máscaras; se reporta la variación de test aparte.

Procedencia / respaldo: CNN-B: historiales.json CPU01–05 y epocas.csv GPU01–05
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 22 -->
6 · FC-A y FC-B: curvas comparables
GRÁFICA NATIVA: LINE_MARKERS (65)
CPU: [7.586207240819931, 15.402299165725708, 21.149425208568573, 25.40229856967926, 28.16092073917389, 31.915709376335144, 33.40996205806732, 37.43295073509216, 38.16092014312744, 43.56321692466736, 45.095786452293396, 47.16475009918213, 48.50574731826782, 50.613027811050415, 52.91187763214111]
CUDA: [7.586207240819931, 18.160919845104218, 22.720307111740112, 24.59770143032074, 28.812259435653687, 30.459770560264587, 35.86207032203674, 40.613025426864624, 40.689656138420105, 43.065133690834045, 47.24137783050537, 46.74329459667206, 49.27203059196472, 48.160919547080994, 49.27203059196472]
GRÁFICA NATIVA: LINE_MARKERS (65)
CPU: [3.2913665771484375, 3.0917515754699707, 2.8379244804382324, 2.584153652191162, 2.4021661281585693, 2.278627872467041, 2.1646840572357178, 2.0454258918762207, 1.965322732925415, 1.8268053531646729, 1.7391629219055176, 1.6864820718765259, 1.6325489282608032, 1.587450385093689, 1.5362921953201294]
CUDA: [3.2670347690582275, 3.051234483718872, 2.749790668487549, 2.5271878242492676, 2.3656442165374756, 2.312250852584839, 2.1610734462738037, 2.0055863857269287, 1.926896572113037, 1.895586609840393, 1.7522492408752441, 1.7136602401733398, 1.6550549268722534, 1.6590479612350464, 1.6039336919784546]
GRÁFICA NATIVA: LINE_MARKERS (65)
CPU: [5.363984778523445, 11.762452125549316, 15.632183849811554, 19.655172526836395, 20.842911303043365, 25.01915693283081, 28.429117798805237, 30.53639829158783, 32.06896483898163, 36.47509515285492, 36.62835359573364, 39.34865891933441, 39.118772745132446, 41.494253277778625, 44.214558601379395]
CUDA: [4.75095771253109, 8.160919696092606, 13.946360349655151, 17.624521255493164, 23.0268195271492, 22.835248708724976, 27.471265196800232, 28.697317838668823, 31.30268156528473, 35.63218414783478, 35.59387028217316, 38.35248947143555, 40.229883790016174, 38.735631108284, 40.07662832736969]
GRÁFICA NATIVA: LINE_MARKERS (65)
CPU: [3.348174810409546, 3.2416882514953613, 3.0324954986572266, 2.8383285999298096, 2.6961073875427246, 2.552922487258911, 2.4089784622192383, 2.278836250305176, 2.161579132080078, 2.0997698307037354, 2.0545992851257324, 1.925100326538086, 1.8425259590148926, 1.795507788658142, 1.7575657367706299]
CUDA: [3.3247272968292236, 3.245692253112793, 3.0364584922790527, 2.8100357055664062, 2.60569429397583, 2.513597011566162, 2.3594553470611572, 2.2711737155914307, 2.193864583969116, 2.0784318447113037, 2.0321121215820312, 1.9162896871566772, 1.859998345375061, 1.8032516241073608, 1.7751481533050537]
Azul: CPU histórica  ·  Naranja: CUDA nueva  ·  15 épocas fijas; no EarlyStopping
22
NOTAS: Arriba FC-A y abajo FC-B. Cada panel muestra cinco corridas promediadas. Dado que el estado y PCG64 se reinician, las cinco trayectorias CUDA son numéricamente repetibles: esto sirve para medir tiempo bajo condiciones similares, pero no estima variación por inicializaciones independientes. La exactitud predictiva de las FC es menor que la de CNN-B y no se mejoró artificialmente cambiando épocas o semillas después de ver resultados.

Procedencia / respaldo: ampliación FC/epocas-comparacion.csv; curvas agregadas en datos-presentacion.json
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 23 -->
6 · Más rápido no significa más exacto
GRÁFICA NATIVA: COLUMN_CLUSTERED (51)
CPU: [96.88888788223267, 53.0651330947876, 44.29118633270264]
CUDA: [98.62068891525269, 51.724135875701904, 39.96168673038483]
98.62%
test CNN-B CUDA
Modelo con mejor desempeño de estas redes
FC-A: −1.34 pp
FC-B: −4.33 pp
Diferencia vs. media CPU
La comparación usa CPU histórica; no son trayectorias CPU/GPU pareadas
23
NOTAS: Separar la métrica de tiempo de la métrica predictiva. CNN-B CUDA alcanza98.6207% en test. Su CPU histórica tuvo resultados variables y no se puede atribuir la diferencia al hardware. FC-A baja de53.065% a51.724%, yFC-B de44.291% a39.962%. Las diferencias son1.341 y4.330 puntos porcentuales. Datos, estado inicial y pasos coinciden, pero no el orden interno ni las máscaras ni todo el redondeo. La implementación demostró aceleración, no mejora predictiva.

Procedencia / respaldo: Calidad FC y calidad_comparada.csv CNN-B; medias de5
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 24 -->
6 · Por qué funciona y qué lo limita











CPU: archivos
Rutas y sistema de archivos
Decodificar y redimensionar
6 imágenes/tareas a la vez
17.32 s → 8.03 s











GPU: matemática
Matrices y tensores grandes
Muchos cálculos independientes
Numba-CUDA + cuBLAS
1.92× · 3.35× · 4.86×











Límites reales
H2D / D2H y sincronización
Validación todavía en CPU
Menos trabajo por lote en FC
RNG y referencia histórica
Tiempo menor es una medición; la causa microarquitectónica exacta requeriría un profiler
24
NOTAS: Responder las tres preguntas técnicas de la guía. Archivos enCPU: muchas operaciones de sistema y tareas independientes por imagen; los workers explotan esa independencia. Matemática enGPU: muchas operaciones similares y productos matriciales reutilizan paralelismo. Limitaciones: preparación y validación CPU, tamaños de lote, transferencias, sincronizaciones, memoria, coste de lanzamiento y carga histórica no pareada. No afirmar que GPU es siempre superior ni que la ocupaciónSM fue medida.

Procedencia / respaldo: Discusión técnica y desglose medido de las tres redes
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 25 -->
De cada resultado a su evidencia

Datos
SHA-256


Configuración
EXP + semillas


Ejecución
RUN + logs


Resultado
CSV + modelo


Evidencia
Capturas + notebook

Notebook CPU
37 celdas de código  ·  10 imágenes
Conservado sin reentrenar

Notebook CUDA
92 celdas de código  ·  30 imágenes
CNN-B + FC-A + FC-B · salidas reales
Tiempos, gráficas y capturas reutilizados: esta presentación no ejecuta entrenamiento
25
NOTAS: Mostrar la cadena de trazabilidad. Los mismos archivos y tensores están identificados por SHA-256; cada ejecución conserva configuración, semillas, logs, métricas y checkpoint. Los notebooks tienen salidas nativas junto al código:92celdas y30PNG CUDA; CPU37y10. Los intentos fallidos se conservan y la primera FC-A ya guardada se recuperó sin reentrenarla. El informe fue editado después por el usuario y esta presentación no lo cambia.

Procedencia / respaldo: Manifiestos, notebooks finales, configuración y validaciones históricas; presentación usa copias de evidencia
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 26 -->
Tres redes completas, una conclusión
1.92×
CNN-B
48.04% menos tiempo
3.35×
FC-A
70.18% menos tiempo
4.86×
FC-B
79.44% menos tiempo
CPU prepara  →  GPU entrena  →  CPU evalúa
Aceleración real · transferencias incluidas · calidad y límites visibles
26
NOTAS: Cerrar con los tres speedups. El aporte consiste en seleccionar la matemática repetitiva, separar CPU/GPU, conservar condiciones de carga comparables y medir los costes completos. Preparación y validaciónCPU no desaparecen. No se declara aceleración del pipeline adaptativo original ni superioridad predictiva. Las conclusiones son locales a estos datos, hardware y protocolo. Abrir preguntas y usar los anexos para los detalles.

Procedencia / respaldo: Comparación final n=5; límites documentados
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 27 -->
A1 · Hardware, librerías y memoria
Componente | Configuración medida
CPU | AMD Ryzen 5 4600H · 6 físicos / 12 lógicos
RAM / GPU | 15.42 GiB · GTX 1650 / 4 GiB / sm_75
CUDA / driver | Toolkit 13.1 · driver 591.59
Python / TensorFlow | PCUDA · Python 3.14.3 · TF 2.22.0rc0 (CPU)
Numba / numba-cuda | 0.67.0 / 0.30.4
Matemática GPU | cuBLAS SGEMM FP32 PEDANTIC · un stream
Memoria FC-A / FC-B | 49.46 MiB / 124.65 MiB de buffers propios
Convoluciones CNN-B | Workspace im2col: 576 MiB
Límites CPU | Intra-op 6 · inter-op 1 · BLAS 6
27
NOTAS: Usar como respaldo de entorno. La arena CUDA propia de las FC excluye contexto y allocations internas de bibliotecas. Para CNN-B se observa1,213MiB deVRAM y usaworkspaceim2col576MiB. La GPU es sm75; no se aplicóTF32 ni mixedprecision. Python y versiones provienen del entorno medido, no de información actual de internet.

Procedencia / respaldo: Hardware y configuraciones congeladas; validación FC
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 28 -->
A2 · Las 14 fases de referencia CPU
Fase CPU | Media (s) | DE (s)
preparacion cpu | 17.325 | 0.414
entrenamiento CNN-A | 124.839 | 2.504
evaluacion CNN-A | 0.802 | 0.058
entrenamiento CNN-B | 805.650 | 2.723
evaluacion CNN-B | 2.322 | 0.072
entrenamiento FC-A | 50.716 | 1.265
evaluacion FC-A | 0.379 | 0.006
entrenamiento FC-B | 102.441 | 3.275
evaluacion FC-B | 0.520 | 0.011
pca | 3.963 | 0.046
entrenamiento RF-A | 2.753 | 0.074
validacion RF-A | 0.054 | 0.001
entrenamiento RF-B | 8.025 | 0.187
validacion RF-B | 0.139 | 0.001
Total común: 1,119.928 s · las validaciones anidadas NO se suman de nuevo
28
NOTAS: Son medias de cinco observaciones válidas por fase. Las cuatro fases de entrenamiento contienen su validación anidada. Para la referencia común se excluyen la rama aumentada, inferencia y las tareas que cambian según la arquitectura seleccionada. La suma es1,119.9275623s y no debe mezclarse con el pipeline adaptativo total.

Procedencia / respaldo: EXP-002 resumen.csv y referencia común
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 29 -->
A3 · Test final: ambas variantes usan CPU
Test del modelo | Referencia CPU (s) | Tras fit CUDA (s) | Cociente CPU / variante
CNN-B | 2.3224 | 2.4922 | 0.932×
FC-A | 0.3788 | 0.3924 | 0.965×
FC-B | 0.5201 | 0.5232 | 0.994×

No se implementó inferencia GPU en este alcance
29
NOTAS: El test mide inferencia CPU con los pesos de cada versión. Las pequeñas diferencias no representan speedup de inferenciaGPU. Todas las fases test se ejecutan con2,610imágenes ybatch32. Este cuadro separa qué sí fue acelerado de qué solamente se ejecutó después del entrenamiento CUDA.

Procedencia / respaldo: comparacion/ampliacion-fc/comparacion.csv, fases evaluacion
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 30 -->
A4 · Actividad GPU y preparación CPU
Red | Actividad media¹ | Máxima observada | VRAM máx. (MiB)
CNN-B | 87.15% | 94% | 1,213
FC-A | 31.31% | 50% | 111
FC-B | 40.76% | 56% | 187
Preparación CPU
6 workers · hasta 12 tareas pendientes
17.325 s → 8.034 s  |  2.156×
¹ Media de porcentajes medios por RUN; muestreo 1 s · actividad agregada, no ocupación SM
30
NOTAS: Los promedios se calculan sobre muestras etiquetadas como entrenamiento; incluyen la granularidad de muestreo de1s y no dan ocupación porSM. Las FC usan menosVRAM y menor actividad agregada, pese a mayores speedups, porque su carga y coste relativoCPU difieren. La captura pequeña documenta preparación CPU6workers en la campaña CUDAoriginal. El resto de capturas principales ya se mostró.

Procedencia / respaldo: recursos_resumen CNN-B, recursos-resumen FC y captura preparación original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 31 -->
A5 · Referencias y términos de apoyo

Akash · ASL Alphabet
Kaggle · dataset de imágenes

NVIDIA · CUDA Python / Numba
Jerarquía y kernels explícitos

NVIDIA · cuBLAS 13.1
SGEMM y biblioteca GPU

NVIDIA · nvidia-smi
Monitorización de actividad y VRAM
Término | Significado
H2D / D2H | Host → Device / Device → Host
CNN / FC | Convolucional / totalmente conectada
DE / CV | Desviación estándar / variación relativa
Speedup | Media CPU ÷ media CUDA
pp | Puntos porcentuales
Datos experimentales propios · citas completas y enlaces en notas del presentador
31
NOTAS: Fuentes externas conservadas de la documentación del proyecto. Las referencias técnicas respaldan jerarquíaCUDA,cuBLASy significado de nvidia-smi; no respaldan los valores experimentales, que proceden de las corridas guardadas. Los títulos completos APA7 yURL están en estas notas. Akash.(s.f.).ASL alphabet[Conjunto de datos].Kaggle.https://www.kaggle.com/datasets/grassknoted/asl-alphabet
NVIDIA.(s.f.-a).cuBLAS library(CUDA Toolkit13.1)[Documentación].https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html
NVIDIA.(s.f.-b).NVIDIA System Management Interface[Documentación].https://docs.nvidia.com/deploy/nvidia-smi/index.html
NVIDIA.(s.f.-c).WritingCUDA kernels.EnNumbaCUDA.https://nvidia.github.io/numba-cuda/user/kernels.html

Procedencia / respaldo: Referencias primarias de informe; resultados propios EXP-002/003/011/012
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.