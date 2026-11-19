# Guion de la presentación corta

15diapositivas. Se conserva la versión completa de31sin modificar. Las notas del expositor están también en elPowerPoint;este guion añade el mapa con la versión larga para consultas.

## 1. De CPU a CUDA

Presentar el reconocimiento de señas ASL como caso de aplicación. El proyecto parte del laboratorio original, mide una referencia CPU y traslada el entrenamiento completo de CNN-B, FC-A y FC-B a CUDA Python. Los tres números son speedup de entrenamiento total medido, con transferencias y validación CPU incluidas. No son aceleración de todo el notebook.

Procedencia / respaldo: Comparación ampliación FC, n=5 por modelo; guía Proyecto 3 CUDA
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 1.

## 2. De 87,000 imágenes a una prueba controlada

El dataset ASL Alphabet contiene 29 clases, letras A–Z y las clases del, nothing y space. Se seleccionaron 600 imágenes por clase, 17,400 en total, con particiones estratificadas 70/15/15. Las seis fotografías provienen de archivos reales del manifiesto de entrenamiento; no son imágenes generadas. Se trabajó con archivos almacenados, no con video en vivo.

Procedencia / respaldo: Manifiesto EXP-002; selección y hashes de seis archivos en datos-presentacion.json
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 3.

## 3. 1 · Las operaciones que se repiten

Explicar cómo una imagen termina generando miles de operaciones por capa. Se cargan 17,400 imágenes; cada red procesa 12,180 por época, en 96 lotes, incluyendo uno de 20. Quince épocas son 1,440 actualizaciones Adam y 182,700 presentaciones de imágenes de entrenamiento por modelo. Las 15 validaciones procesan 39,150 imágenes y no actualizan pesos. Estos son los volúmenes de la campaña fija, no el EarlyStopping histórico del notebook original.

Procedencia / respaldo: Análisis original; configuraciones EXP-002/003/012
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 5.

## 4. 2 · Referencia CPU: medir antes de cambiar

La referencia no es estrictamente monohilo. TensorFlow CPU usa seis hilos intra-op y uno inter-op, BLAS seis; Random Forest mantiene n_jobs=-1. El calentamiento y las importaciones quedan fuera; se restaura el estado antes de medir. Las cinco corridas completas válidas son CPU01–05. La carga base fija comparte 14 fases y tiene variación baja. El total adaptativo mezcla distintas CNN aumentadas y no es una referencia homogénea para CUDA.

Procedencia / respaldo: EXP-002: serie.json, resultados.md y resumen_referencia_comun.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 6.

## 5. 2 · CNN-B concentraba el tiempo CPU

Se suman las medias de 14 fases no solapadas. CNN-B representa 71.94% del total común. FC-A y FC-B aportan más tiempo de entrenamiento paralelizable; con las tres se cubre aproximadamente 85.61% de esta referencia. El donut no representa el pipeline adaptativo completo. La validación dentro de fit ya está incluida y no se añade otra vez.

Procedencia / respaldo: CPU EXP-002 resumen.csv; referencia común 1119.9275623 s
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 7.

## 6. 3 · ¿Qué se paralelizó y por qué?

No se trasladó todo el programa al GPU. Se seleccionaron la matemática del entrenamiento de tres redes y sus operaciones por elemento o por reducción. La carga de archivos queda en CPU porque implica rutas, sistema de archivos, decodificación e I/O irregular. PCA, Random Forest, CNN-A y augmentation no se portaron en este alcance. La validación y test siguen en CPU por el diseño aplicado; sus costes se declaran.

Procedencia / respaldo: Paso3, implementación revisada y ampliación FC
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 8.

## 7. 4 · Un flujo CPU ↔ GPU explícito

Recorrer la separación de fases. CPU realiza lectura y preparación, genera el orden de lotes y máscaras y usa memoria host fijada para copiar. H2D envía datos al stream. GPU hace forward, pérdida, backward y Adam. Al final de cada época, D2H exporta pesos y estadísticas BN para validar en CPU. El ciclo se repite quince veces. Preparación no se presenta como CUDA y test no se presenta como inferencia GPU.

Procedencia / respaldo: Diseño aplicado: modelo Network/FCNetwork, un stream, sin solapamiento
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 9.

## 8. 4 · Threads, blocks y grids

Explicar los tres niveles CUDA con el dibujo. Los kernels propios distribuyen elementos en hilos de bloques de 256 y usan recorrido grid-stride. La reducción parcial BN usa 32×8; la final FC asigna un hilo por característica. Softmax usa 32 hilos y un bloque por muestra. Los productos SGEMM se delegan a cuBLAS: su reparto interno es de NVIDIA y no se atribuye al equipo.

Procedencia / respaldo: Código CUDA medido; recursos compilados y lanzamientos por RUN
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 11.

## 9. 5 · Qué se implementó y qué se midió

La implementación usa Numba-CUDA para kernels Python y cuBLAS SGEMM para los productos grandes; TensorFlow/Keras está restringido a CPU y espejo para validar/guardar. El intervalo principal es pared sincronizada y contiene staging/máscaras, H2D, forward/backward/Adam, D2H y validación CPU. Preparación de imágenes, warmup/compilación y guardado/figuras se separan. Un único stream mantiene el orden y no hay solapamiento.

Procedencia / respaldo: EXP-003 y colección FC EXP-011/012; fuentes congeladas y temporizadores reales
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 12.

## 10. 5 · CPU real durante su fase de trabajo

Esta captura pertenece a la ejecución adicional n=1 del notebook CPU, separada de los promedios de la campaña. Fue tomada durante entrenamiento CNN-B en CPU, no durante preparación. psutil usa 100% por procesador lógico, por eso 507.3% indica varios núcleos activos. GPU estaba a0%. La evidencia de preparación con seis workers se presenta en el respaldo de recursos; no atribuir esta captura a I/O.

Procedencia / respaldo: EXP-008 captura-cpu-sostenido.png y JSON de muestra original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 14.

## 11. 5 · GPU real durante entrenamiento CNN-B

La captura es de la corrida adicional de notebook CUDA y no modifica las cinco medias previas. nvidia-smi registró91% de actividad GPU y1,213MiB de4,096, un29.6% de capacidad VRAM. Actividad agregada no significa91% de núcleos ni ocupación por SM. Es evidencia de uso real y actividad elevada; no prueba saturación total de memoria.

Procedencia / respaldo: EXP-009 captura-cuda.png y metadata original
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 15.

## 12. 6 · Tres redes, menos tiempo total

Esta es la tabla principal de resultados. Cada cifra resume cinco corridas por dispositivo/modelo. CUDA incluye transferencias y validación CPU. Los tres minigráficos tienen escalas verticales diferentes: comparan CPU/GPU dentro de cada arquitectura. Mostrar la fórmula speedup como cociente entre medias, no media de cocientes y no suma de speedups. La CPU es histórica; no se repitió la campaña para preparar la ampliación.

Procedencia / respaldo: comparacion/ampliacion-fc/comparacion.csv y estadisticas.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 17.

## 13. 6 · ¿Dónde se va el tiempo con CUDA?

El gráfico normaliza cada entrenamiento a100% para comparar proporciones; los tiempos absolutos de las tres redes son muy diferentes. CNN-B pasa la mayor parte del intervalo en matemática del stream. En las FC crecen las proporciones de validación CPU y administración. H2D/D2H son costes incluidos. Los eventos del stream pueden contener huecos de envío del host; no significan tiempo activo exclusivo de un kernel. La tabla muestra segundos para evitar confundir porcentaje con tiempo absoluto.

Procedencia / respaldo: Desglose CNN-B paso6 y ampliación FC/desglose.csv
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 20.

## 14. 6 · Más rápido no significa más exacto

Separar la métrica de tiempo de la métrica predictiva. CNN-B CUDA alcanza98.6207% en test. Su CPU histórica tuvo resultados variables y no se puede atribuir la diferencia al hardware. FC-A baja de53.065% a51.724%, yFC-B de44.291% a39.962%. Las diferencias son1.341 y4.330 puntos porcentuales. Datos, estado inicial y pasos coinciden, pero no el orden interno ni las máscaras ni todo el redondeo. La implementación demostró aceleración, no mejora predictiva.

Procedencia / respaldo: Calidad FC y calidad_comparada.csv CNN-B; medias de5
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 23.

## 15. Tres redes completas, una conclusión

Cerrar con los tres speedups. El aporte consiste en seleccionar la matemática repetitiva, separar CPU/GPU, conservar condiciones de carga comparables y medir los costes completos. Preparación y validaciónCPU no desaparecen. No se declara aceleración del pipeline adaptativo original ni superioridad predictiva. Las conclusiones son locales a estos datos, hardware y protocolo. Abrir preguntas y consultar la presentación completa para los detalles.

Procedencia / respaldo: Comparación final n=5; límites documentados
Las cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.

Diapositiva de referencia en versión completa: 26.
