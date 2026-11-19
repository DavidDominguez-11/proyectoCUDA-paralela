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
2
NOTAS: El dataset ASL Alphabet contiene 29 clases, letras A–Z y las clases del, nothing y space. Se seleccionaron 600 imágenes por clase, 17,400 en total, con particiones estratificadas 70/15/15. Las seis fotografías provienen de archivos reales del manifiesto de entrenamiento; no son imágenes generadas. Se trabajó con archivos almacenados, no con video en vivo.

Procedencia / respaldo: Manifiesto EXP-002; selección y hashes de seis archivos en datos-presentacion.json
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 3 -->
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
3
NOTAS: Explicar cómo una imagen termina generando miles de operaciones por capa. Se cargan 17,400 imágenes; cada red procesa 12,180 por época, en 96 lotes, incluyendo uno de 20. Quince épocas son 1,440 actualizaciones Adam y 182,700 presentaciones de imágenes de entrenamiento por modelo. Las 15 validaciones procesan 39,150 imágenes y no actualizan pesos. Estos son los volúmenes de la campaña fija, no el EarlyStopping histórico del notebook original.

Procedencia / respaldo: Análisis original; configuraciones EXP-002/003/012
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 4 -->
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
4
NOTAS: La referencia no es estrictamente monohilo. TensorFlow CPU usa seis hilos intra-op y uno inter-op, BLAS seis; Random Forest mantiene n_jobs=-1. El calentamiento y las importaciones quedan fuera; se restaura el estado antes de medir. Las cinco corridas completas válidas son CPU01–05. La carga base fija comparte 14 fases y tiene variación baja. El total adaptativo mezcla distintas CNN aumentadas y no es una referencia homogénea para CUDA.

Procedencia / respaldo: EXP-002: serie.json, resultados.md y resumen_referencia_comun.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 5 -->
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
5
NOTAS: Se suman las medias de 14 fases no solapadas. CNN-B representa 71.94% del total común. FC-A y FC-B aportan más tiempo de entrenamiento paralelizable; con las tres se cubre aproximadamente 85.61% de esta referencia. El donut no representa el pipeline adaptativo completo. La validación dentro de fit ya está incluida y no se añade otra vez.

Procedencia / respaldo: CPU EXP-002 resumen.csv; referencia común 1119.9275623 s
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 6 -->
3 · ¿Qué se paralelizó y por qué?
Operación | ¿Qué puede ocurrir a la vez? | Ejecutor final
Archivos / decodificación / resize | Imágenes independientes | CPU · 6 workers
Convoluciones / Dense | Productos y sumas de matrices | GPU · cuBLAS
ReLU / Dropout / normalización | Elementos y características del lote | GPU · kernels Python
Pérdida / gradientes / Adam | Muestras, parámetros y reducciones | GPU · kernels Python
Validación / test | Predicciones del modelo exportado | CPU
PCA / RF / CNN-A / augmentation | Fuera del alcance CUDA elegido | CPU original

CNN-B + FC-A + FC-B: entrenamiento completo en GPU
6
NOTAS: No se trasladó todo el programa al GPU. Se seleccionaron la matemática del entrenamiento de tres redes y sus operaciones por elemento o por reducción. La carga de archivos queda en CPU porque implica rutas, sistema de archivos, decodificación e I/O irregular. PCA, Random Forest, CNN-A y augmentation no se portaron en este alcance. La validación y test siguen en CPU por el diseño aplicado; sus costes se declaran.

Procedencia / respaldo: Paso3, implementación revisada y ampliación FC
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 7 -->
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
7
NOTAS: Recorrer la separación de fases. CPU realiza lectura y preparación, genera el orden de lotes y máscaras y usa memoria host fijada para copiar. H2D envía datos al stream. GPU hace forward, pérdida, backward y Adam. Al final de cada época, D2H exporta pesos y estadísticas BN para validar en CPU. El ciclo se repite quince veces. Preparación no se presenta como CUDA y test no se presenta como inferencia GPU.

Procedencia / respaldo: Diseño aplicado: modelo Network/FCNetwork, un stream, sin solapamiento
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 8 -->
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
8
NOTAS: Explicar los tres niveles CUDA con el dibujo. Los kernels propios distribuyen elementos en hilos de bloques de 256 y usan recorrido grid-stride. La reducción parcial BN usa 32×8; la final FC asigna un hilo por característica. Softmax usa 32 hilos y un bloque por muestra. Los productos SGEMM se delegan a cuBLAS: su reparto interno es de NVIDIA y no se atribuye al equipo.

Procedencia / respaldo: Código CUDA medido; recursos compilados y lanzamientos por RUN
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 9 -->
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
9
NOTAS: La implementación usa Numba-CUDA para kernels Python y cuBLAS SGEMM para los productos grandes; TensorFlow/Keras está restringido a CPU y espejo para validar/guardar. El intervalo principal es pared sincronizada y contiene staging/máscaras, H2D, forward/backward/Adam, D2H y validación CPU. Preparación de imágenes, warmup/compilación y guardado/figuras se separan. Un único stream mantiene el orden y no hay solapamiento.

Procedencia / respaldo: EXP-003 y colección FC EXP-011/012; fuentes congeladas y temporizadores reales
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 10 -->
5 · CPU real durante su fase de trabajo
507.3%
CPU proceso + hijos
100% equivale a un núcleo
0%
GPU en esta muestra
Captura durante CPU
Monitor en vivo · lectura original psutil / nvidia-smi · muestra sostenida, época 3
10
NOTAS: Esta captura pertenece a la ejecución adicional n=1 del notebook CPU, separada de los promedios de la campaña. Fue tomada durante entrenamiento CNN-B en CPU, no durante preparación. psutil usa 100% por procesador lógico, por eso 507.3% indica varios núcleos activos. GPU estaba a0%. La evidencia de preparación con seis workers se presenta en el respaldo de recursos; no atribuir esta captura a I/O.

Procedencia / respaldo: EXP-008 captura-cpu-sostenido.png y JSON de muestra original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 11 -->
5 · GPU real durante entrenamiento CNN-B
91%
actividad GPU
Agregado nvidia-smi
1,213
MiB de VRAM
29.6% de los 4,096 MiB
Actividad GPU ≠ ocupación por SM; VRAM utilizada ≠ saturación de toda la memoria
11
NOTAS: La captura es de la corrida adicional de notebook CUDA y no modifica las cinco medias previas. nvidia-smi registró91% de actividad GPU y1,213MiB de4,096, un29.6% de capacidad VRAM. Actividad agregada no significa91% de núcleos ni ocupación por SM. Es evidencia de uso real y actividad elevada; no prueba saturación total de memoria.

Procedencia / respaldo: EXP-009 captura-cuda.png y metadata original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 12 -->
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
12
NOTAS: Esta es la tabla principal de resultados. Cada cifra resume cinco corridas por dispositivo/modelo. CUDA incluye transferencias y validación CPU. Los tres minigráficos tienen escalas verticales diferentes: comparan CPU/GPU dentro de cada arquitectura. Mostrar la fórmula speedup como cociente entre medias, no media de cocientes y no suma de speedups. La CPU es histórica; no se repitió la campaña para preparar la ampliación.

Procedencia / respaldo: comparacion/ampliacion-fc/comparacion.csv y estadisticas.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 13 -->
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
13
NOTAS: El gráfico normaliza cada entrenamiento a100% para comparar proporciones; los tiempos absolutos de las tres redes son muy diferentes. CNN-B pasa la mayor parte del intervalo en matemática del stream. En las FC crecen las proporciones de validación CPU y administración. H2D/D2H son costes incluidos. Los eventos del stream pueden contener huecos de envío del host; no significan tiempo activo exclusivo de un kernel. La tabla muestra segundos para evitar confundir porcentaje con tiempo absoluto.

Procedencia / respaldo: Desglose CNN-B paso6 y ampliación FC/desglose.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 14 -->
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
14
NOTAS: Separar la métrica de tiempo de la métrica predictiva. CNN-B CUDA alcanza98.6207% en test. Su CPU histórica tuvo resultados variables y no se puede atribuir la diferencia al hardware. FC-A baja de53.065% a51.724%, yFC-B de44.291% a39.962%. Las diferencias son1.341 y4.330 puntos porcentuales. Datos, estado inicial y pasos coinciden, pero no el orden interno ni las máscaras ni todo el redondeo. La implementación demostró aceleración, no mejora predictiva.

Procedencia / respaldo: Calidad FC y calidad_comparada.csv CNN-B; medias de5
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

<!-- Slide number: 15 -->
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
15
NOTAS: Cerrar con los tres speedups. El aporte consiste en seleccionar la matemática repetitiva, separar CPU/GPU, conservar condiciones de carga comparables y medir los costes completos. Preparación y validaciónCPU no desaparecen. No se declara aceleración del pipeline adaptativo original ni superioridad predictiva. Las conclusiones son locales a estos datos, hardware y protocolo. Abrir preguntas y usar los anexos para los detalles.

Procedencia / respaldo: Comparación final n=5; límites documentados
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.