# Guion de exposición

Las diapositivas 1–26 forman el recorrido principal; 27–31 son respaldo para preguntas. Cada diapositiva contiene notas del presentador en PowerPoint. Se pueden abrir sin ejecutar modelos.

## 1. De CPU a CUDA

Presentar el reconocimiento de señas ASL como caso de aplicación. El proyecto parte del laboratorio original, mide una referencia CPU y traslada el entrenamiento completo de CNN-B, FC-A y FC-B a CUDA Python. Los tres números son speedup de entrenamiento total medido, con transferencias y validación CPU incluidas. No son aceleración de todo el notebook.

Procedencia: Comparación ampliación FC, n=5 por modelo; guía Proyecto 3 CUDA

## 2. El proyecto en seis pasos

Seguir el mismo orden que la guía. Los diagramas resumen el procedimiento; después se concentra la exposición en tiempos, variabilidad, calidad y evidencia de recursos. La extensión FC está integrada al diseño y a las conclusiones finales.

Procedencia: Guía principal, apartados 1–6

## 3. De 87,000 imágenes a una prueba controlada

El dataset ASL Alphabet contiene 29 clases, letras A–Z y las clases del, nothing y space. Se seleccionaron 600 imágenes por clase, 17,400 en total, con particiones estratificadas 70/15/15. Las seis fotografías provienen de archivos reales del manifiesto de entrenamiento; no son imágenes generadas. Se trabajó con archivos almacenados, no con video en vivo.

Procedencia: Manifiesto EXP-002; selección y hashes de seis archivos en datos-presentacion.json

## 4. 1 · ¿Qué hacía el programa original?

Describir la entrada de imágenes, la selección y transformación a tensores. El laboratorio entrena cuatro redes base y una variante con aumento, además de PCA y Random Forest. Después compara, guarda e infiere sobre fotos del equipo. La rama aumentada cambia según la selección: por eso su total adaptativo no se usó como denominador de speedup. El original se conserva intacto.

Procedencia: Notebook original y análisis original; CPU EXP-002

## 5. 1 · Las operaciones que se repiten

Explicar cómo una imagen termina generando miles de operaciones por capa. Se cargan 17,400 imágenes; cada red procesa 12,180 por época, en 96 lotes, incluyendo uno de 20. Quince épocas son 1,440 actualizaciones Adam y 182,700 presentaciones de imágenes de entrenamiento por modelo. Las 15 validaciones procesan 39,150 imágenes y no actualizan pesos. Estos son los volúmenes de la campaña fija, no el EarlyStopping histórico del notebook original.

Procedencia: Análisis original; configuraciones EXP-002/003/012

## 6. 2 · Referencia CPU: medir antes de cambiar

La referencia no es estrictamente monohilo. TensorFlow CPU usa seis hilos intra-op y uno inter-op, BLAS seis; Random Forest mantiene n_jobs=-1. El calentamiento y las importaciones quedan fuera; se restaura el estado antes de medir. Las cinco corridas completas válidas son CPU01–05. La carga base fija comparte 14 fases y tiene variación baja. El total adaptativo mezcla distintas CNN aumentadas y no es una referencia homogénea para CUDA.

Procedencia: EXP-002: serie.json, resultados.md y resumen_referencia_comun.csv

## 7. 2 · CNN-B concentraba el tiempo CPU

Se suman las medias de 14 fases no solapadas. CNN-B representa 71.94% del total común. FC-A y FC-B aportan más tiempo de entrenamiento paralelizable; con las tres se cubre aproximadamente 85.61% de esta referencia. El donut no representa el pipeline adaptativo completo. La validación dentro de fit ya está incluida y no se añade otra vez.

Procedencia: CPU EXP-002 resumen.csv; referencia común 1119.9275623 s

## 8. 3 · ¿Qué se paralelizó y por qué?

No se trasladó todo el programa al GPU. Se seleccionaron la matemática del entrenamiento de tres redes y sus operaciones por elemento o por reducción. La carga de archivos queda en CPU porque implica rutas, sistema de archivos, decodificación e I/O irregular. PCA, Random Forest, CNN-A y augmentation no se portaron en este alcance. La validación y test siguen en CPU por el diseño aplicado; sus costes se declaran.

Procedencia: Paso3, implementación revisada y ampliación FC

## 9. 4 · Un flujo CPU ↔ GPU explícito

Recorrer la separación de fases. CPU realiza lectura y preparación, genera el orden de lotes y máscaras y usa memoria host fijada para copiar. H2D envía datos al stream. GPU hace forward, pérdida, backward y Adam. Al final de cada época, D2H exporta pesos y estadísticas BN para validar en CPU. El ciclo se repite quince veces. Preparación no se presenta como CUDA y test no se presenta como inferencia GPU.

Procedencia: Diseño aplicado: modelo Network/FCNetwork, un stream, sin solapamiento

## 10. 4 · Tres redes, la misma regla de reparto

FC significa fully connected: red de capas densas, diferente de la CNN que aprende filtros espaciales. CNN-B contiene cinco convoluciones con 32/32/64/64/128 filtros, tres pooling y Dense256. Las FC aplanan los 12,288 valores de la imagen y usan BN al inicio. Se preservan las arquitecturas, incluyendo Dropout y la diferencia de momentum BN. Las cifras son parámetros entrenables, no el número de operaciones ni la VRAM observada.

Procedencia: Construcciones originales + FCNetwork; inicializaciones históricas preservadas

## 11. 4 · Threads, blocks y grids

Explicar los tres niveles CUDA con el dibujo. Los kernels propios distribuyen elementos en hilos de bloques de 256 y usan recorrido grid-stride. La reducción parcial BN usa 32×8; la final FC asigna un hilo por característica. Softmax usa 32 hilos y un bloque por muestra. Los productos SGEMM se delegan a cuBLAS: su reparto interno es de NVIDIA y no se atribuye al equipo.

Procedencia: Código CUDA medido; recursos compilados y lanzamientos por RUN

## 12. 5 · Qué se implementó y qué se midió

La implementación usa Numba-CUDA para kernels Python y cuBLAS SGEMM para los productos grandes; TensorFlow/Keras está restringido a CPU y espejo para validar/guardar. El intervalo principal es pared sincronizada y contiene staging/máscaras, H2D, forward/backward/Adam, D2H y validación CPU. Preparación de imágenes, warmup/compilación y guardado/figuras se separan. Un único stream mantiene el orden y no hay solapamiento.

Procedencia: EXP-003 y colección FC EXP-011/012; fuentes congeladas y temporizadores reales

## 13. 5 · Verificar antes de confiar en el tiempo

Las pruebas locales contrastan operadores y gradientes con un oráculo CPU. Se verificó la restauración tras warmup y el contador Adam real de los checkpoints. La corrección local aprobada no implica una trayectoria numérica idéntica durante quince épocas: RNG, orden y redondeo difieren y en la CNN hubo diagnósticos estrictos de rutas cerca de empates que no se etiquetaron como aprobados. Se conserva esa reserva.

Procedencia: Pruebas operadores/adjuntos CNN-B; validacion-01 FC; auditoría HDF5 final

## 14. 5 · CPU real durante su fase de trabajo

Esta captura pertenece a la ejecución adicional n=1 del notebook CPU, separada de los promedios de la campaña. Fue tomada durante entrenamiento CNN-B en CPU, no durante preparación. psutil usa 100% por procesador lógico, por eso 507.3% indica varios núcleos activos. GPU estaba a0%. La evidencia de preparación con seis workers se presenta en el respaldo de recursos; no atribuir esta captura a I/O.

Procedencia: EXP-008 captura-cpu-sostenido.png y JSON de muestra original

## 15. 5 · GPU real durante entrenamiento CNN-B

La captura es de la corrida adicional de notebook CUDA y no modifica las cinco medias previas. nvidia-smi registró91% de actividad GPU y1,213MiB de4,096, un29.6% de capacidad VRAM. Actividad agregada no significa91% de núcleos ni ocupación por SM. Es evidencia de uso real y actividad elevada; no prueba saturación total de memoria.

Procedencia: EXP-009 captura-cuda.png y metadata original

## 16. 5 · FC-A y FC-B también usaron CUDA

Son capturas originales durante la primera corrida válida de cada FC. FC-A registró50% de actividad y111MiB; FC-B32% y187MiB en esas muestras. Las capturas no son máximos globales. Una GPU menos activa puede terminar antes cuando tiene menos trabajo por lote o mayores proporciones de host y validación; el speedup se determina por tiempo total. No se llenó VRAM artificialmente.

Procedencia: EXP-011 FC-A1 y EXP-012 FC-B1: capturas y CSV originales

## 17. 6 · Tres redes, menos tiempo total

Esta es la tabla principal de resultados. Cada cifra resume cinco corridas por dispositivo/modelo. CUDA incluye transferencias y validación CPU. Los tres minigráficos tienen escalas verticales diferentes: comparan CPU/GPU dentro de cada arquitectura. Mostrar la fórmula speedup como cociente entre medias, no media de cocientes y no suma de speedups. La CPU es histórica; no se repitió la campaña para preparar la ampliación.

Procedencia: comparacion/ampliacion-fc/comparacion.csv y estadisticas.csv

## 18. 6 · Las cinco repeticiones, sin escoger la mejor

Mostrar todas las observaciones de entrenamiento. No se eligió solo la ejecución más rápida. FC-A1 guardada en EXP-011 se reutilizó después de corregir solo la presentación; EXP-012 hizo las nueve pendientes. Un intento FC-A incompleto de EXP-010 está excluido del promedio por no cerrar/checkpointear, no por su velocidad. Las cinco repeticiones CUDA parten del mismo estado y RNG: la variación resume tiempo, no cinco aprendizajes independientes.

Procedencia: observaciones.csv; series CPU01–05, GPU01–05, FC colección EXP-012

## 19. 6 · Comparar fases sin mezclar causas

La preparación mejoró de17.325s a8.034s con seis workers CPU, cociente2.156x: no es aceleración CUDA. El test final continúa CPU y no mejoró. La suma comparable de preparación, entrenamiento CNN-B y test pasa825.297 a429.134s:1.923x, pero no es el pipeline completo. Como cálculo aparte, sumar los tres entrenamientos da958.806 a454.792s y2.108x, sin medir un pipeline continuo ni añadir los cocientes individuales.

Procedencia: paso-6-revision-02/comparacion_fases.csv y tres medias de fit

## 20. 6 · ¿Dónde se va el tiempo con CUDA?

El gráfico normaliza cada entrenamiento a100% para comparar proporciones; los tiempos absolutos de las tres redes son muy diferentes. CNN-B pasa la mayor parte del intervalo en matemática del stream. En las FC crecen las proporciones de validación CPU y administración. H2D/D2H son costes incluidos. Los eventos del stream pueden contener huecos de envío del host; no significan tiempo activo exclusivo de un kernel. La tabla muestra segundos para evitar confundir porcentaje con tiempo absoluto.

Procedencia: Desglose CNN-B paso6 y ampliación FC/desglose.csv

## 21. 6 · CNN-B: aprendizaje durante 15 épocas

Los trazos son medias de cinco trayectorias por implementación y se calculan a partir de historiales guardados. AzulCPU, naranjaCUDA. La validación se ejecuta en CPU en ambas variantes, con los pesos exportados cuando el entrenamiento es CUDA. Los resultados no son una ejecución adicional. Curvas parecidas no prueban identidad de gradientes ni de máscaras; se reporta la variación de test aparte.

Procedencia: CNN-B: historiales.json CPU01–05 y epocas.csv GPU01–05

## 22. 6 · FC-A y FC-B: curvas comparables

Arriba FC-A y abajo FC-B. Cada panel muestra cinco corridas promediadas. Dado que el estado y PCG64 se reinician, las cinco trayectorias CUDA son numéricamente repetibles: esto sirve para medir tiempo bajo condiciones similares, pero no estima variación por inicializaciones independientes. La exactitud predictiva de las FC es menor que la de CNN-B y no se mejoró artificialmente cambiando épocas o semillas después de ver resultados.

Procedencia: ampliación FC/epocas-comparacion.csv; curvas agregadas en datos-presentacion.json

## 23. 6 · Más rápido no significa más exacto

Separar la métrica de tiempo de la métrica predictiva. CNN-B CUDA alcanza98.6207% en test. Su CPU histórica tuvo resultados variables y no se puede atribuir la diferencia al hardware. FC-A baja de53.065% a51.724%, yFC-B de44.291% a39.962%. Las diferencias son1.341 y4.330 puntos porcentuales. Datos, estado inicial y pasos coinciden, pero no el orden interno ni las máscaras ni todo el redondeo. La implementación demostró aceleración, no mejora predictiva.

Procedencia: Calidad FC y calidad_comparada.csv CNN-B; medias de5

## 24. 6 · Por qué funciona y qué lo limita

Responder las tres preguntas técnicas de la guía. Archivos enCPU: muchas operaciones de sistema y tareas independientes por imagen; los workers explotan esa independencia. Matemática enGPU: muchas operaciones similares y productos matriciales reutilizan paralelismo. Limitaciones: preparación y validación CPU, tamaños de lote, transferencias, sincronizaciones, memoria, coste de lanzamiento y carga histórica no pareada. No afirmar que GPU es siempre superior ni que la ocupaciónSM fue medida.

Procedencia: Discusión técnica y desglose medido de las tres redes

## 25. De cada resultado a su evidencia

Mostrar la cadena de trazabilidad. Los mismos archivos y tensores están identificados por SHA-256; cada ejecución conserva configuración, semillas, logs, métricas y checkpoint. Los notebooks tienen salidas nativas junto al código:92celdas y30PNG CUDA; CPU37y10. Los intentos fallidos se conservan y la primera FC-A ya guardada se recuperó sin reentrenarla. El informe fue editado después por el usuario y esta presentación no lo cambia.

Procedencia: Manifiestos, notebooks finales, configuración y validaciones históricas; presentación usa copias de evidencia

## 26. Tres redes completas, una conclusión

Cerrar con los tres speedups. El aporte consiste en seleccionar la matemática repetitiva, separar CPU/GPU, conservar condiciones de carga comparables y medir los costes completos. Preparación y validaciónCPU no desaparecen. No se declara aceleración del pipeline adaptativo original ni superioridad predictiva. Las conclusiones son locales a estos datos, hardware y protocolo. Abrir preguntas y usar los anexos para los detalles.

Procedencia: Comparación final n=5; límites documentados

## 27. A1 · Hardware, librerías y memoria

Usar como respaldo de entorno. La arena CUDA propia de las FC excluye contexto y allocations internas de bibliotecas. Para CNN-B se observa1,213MiB deVRAM y usaworkspaceim2col576MiB. La GPU es sm75; no se aplicóTF32 ni mixedprecision. Python y versiones provienen del entorno medido, no de información actual de internet.

Procedencia: Hardware y configuraciones congeladas; validación FC

## 28. A2 · Las 14 fases de referencia CPU

Son medias de cinco observaciones válidas por fase. Las cuatro fases de entrenamiento contienen su validación anidada. Para la referencia común se excluyen la rama aumentada, inferencia y las tareas que cambian según la arquitectura seleccionada. La suma es1,119.9275623s y no debe mezclarse con el pipeline adaptativo total.

Procedencia: EXP-002 resumen.csv y referencia común

## 29. A3 · Test final: ambas variantes usan CPU

El test mide inferencia CPU con los pesos de cada versión. Las pequeñas diferencias no representan speedup de inferenciaGPU. Todas las fases test se ejecutan con2,610imágenes ybatch32. Este cuadro separa qué sí fue acelerado de qué solamente se ejecutó después del entrenamiento CUDA.

Procedencia: comparacion/ampliacion-fc/comparacion.csv, fases evaluacion

## 30. A4 · Actividad GPU y preparación CPU

Los promedios se calculan sobre muestras etiquetadas como entrenamiento; incluyen la granularidad de muestreo de1s y no dan ocupación porSM. Las FC usan menosVRAM y menor actividad agregada, pese a mayores speedups, porque su carga y coste relativoCPU difieren. La captura pequeña documenta preparación CPU6workers en la campaña CUDAoriginal. El resto de capturas principales ya se mostró.

Procedencia: recursos_resumen CNN-B, recursos-resumen FC y captura preparación original

## 31. A5 · Referencias y términos de apoyo

Fuentes externas conservadas de la documentación del proyecto. Las referencias técnicas respaldan jerarquíaCUDA,cuBLASy significado de nvidia-smi; no respaldan los valores experimentales, que proceden de las corridas guardadas. Los títulos completos APA7 yURL están en estas notas. Akash.(s.f.).ASL alphabet[Conjunto de datos].Kaggle.https://www.kaggle.com/datasets/grassknoted/asl-alphabet
NVIDIA.(s.f.-a).cuBLAS library(CUDA Toolkit13.1)[Documentación].https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html
NVIDIA.(s.f.-b).NVIDIA System Management Interface[Documentación].https://docs.nvidia.com/deploy/nvidia-smi/index.html
NVIDIA.(s.f.-c).WritingCUDA kernels.EnNumbaCUDA.https://nvidia.github.io/numba-cuda/user/kernels.html

Procedencia: Referencias primarias de informe; resultados propios EXP-002/003/011/012
