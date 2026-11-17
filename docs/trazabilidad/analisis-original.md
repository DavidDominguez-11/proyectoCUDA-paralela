# Paso 1. Análisis de la solución secuencial existente

Estado: **paso 1 completado mediante revisión estática**, conforme a las páginas
1 y 2 del [PDF oficial](../Proyecto%203%20CUDA.pdf). Relación: REQ-02,
PROC-01/02, DEC-010, CAM-002 y BIT-004.

Tipo de evidencia: inspección estática del
[notebook original](../../LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb) y lectura
de sus salidas históricas. Se numeran las celdas desde cero; el número es una
referencia inicial, pues puede cambiar al editar. La identidad exacta del archivo
está en [el manifiesto inicial](../../resultados/inventario-inicial.csv).

La [evidencia de lectura](../../resultados/analisis-original/revision-estatica.md)
conserva las 33 celdas de código, sus hashes y las salidas históricas relevantes
del notebook de 68 celdas. No se ejecutó el algoritmo para completar este paso.

## Propósito y principales partes

El programa clasifica imágenes de manos en 29 clases: A–Z, `del`, `nothing` y
`space`. Recibe archivos almacenados, no video ni capturas de cámara en vivo.
Su salida es una clase predicha por imagen, junto con métricas y visualizaciones
que comparan modelos y muestran aciertos/errores.

Se divide en cinco partes principales:

1. **Adquisición y exploración:** obtiene el dataset, identifica sus clases y
   examina cantidad, formato y variabilidad de las imágenes.
2. **Preparación:** selecciona archivos, separa train/val/test, convierte imágenes
   en tensores normalizados y codifica sus etiquetas.
3. **Construcción y entrenamiento:** ajusta CNN, redes densas y PCA/Random Forest;
   prueba además una CNN con aumento de imágenes.
4. **Evaluación y selección:** calcula pérdida, accuracy y matriz de confusión,
   compara modelos y persiste el seleccionado y otros artefactos.
5. **Inferencia con fotos del equipo:** recarga modelos, prepara imágenes nuevas,
   obtiene etiquetas y presenta resultados individuales y agregados.

Se denomina «solución secuencial existente» por su papel como programa de partida
en la consigna. El notebook no fuerza un único hilo: Random Forest configura
`n_jobs=-1`, y el paralelismo interno de las librerías no está restringido. Tampoco
fuerza explícitamente un dispositivo para Keras. Esto no constituye todavía una
referencia CPU controlada ni una implementación CUDA del proyecto.

## Tabla 1. Descomposición de operaciones

Las referencias a celdas y funciones forman parte de la descripción para poder
contrastar cada fila con el código original. El orden sigue el notebook completo.

| No. | Operación identificada | Descripción de su funcionamiento |
|---|---|---|
| 1 | Configuración e importación de librerías | Carga NumPy, pandas, Pillow, sklearn y herramientas gráficas; fija `SEED=42` para Python/NumPy y configura las gráficas. TensorFlow/Keras se importa al construir las redes. Celdas 2 y 31. |
| 2 | Recepción del dataset de imágenes | `kagglehub.dataset_download` descarga o recupera desde caché `grassknoted/asl-alphabet`; construye rutas de entrenamiento/prueba oficial y comprueba que existan. Celda 4. |
| 3 | Identificación de clases e inventario | Ordena alfabéticamente las carpetas de clases y cuenta archivos `.jpg` por clase. Las salidas guardadas registran 29 clases, 87,000 imágenes disponibles y 28 imágenes de prueba oficial. Celdas 4 y 6. |
| 4 | Exploración visual y de metadatos | `show_class_samples` y `show_confusable_groups` muestran variabilidad y letras parecidas; otra muestra permite revisar tamaño, modo RGB y formato JPEG. La gráfica de conteos permite observar balance de clases. Celdas 9, 12, 15 y 18. |
| 5 | Selección de la submuestra | `build_sampled_file_list` recorre las clases y selecciona hasta 600 archivos por cada una con una semilla fija. Produce un DataFrame con ruta y etiqueta; el conjunto histórico contiene 17,400 imágenes. Celda 22. |
| 6 | División de datos | Dos llamadas a `train_test_split` generan particiones estratificadas 70/15/15: 12,180 imágenes para entrenamiento, 2,610 para validación y 2,610 para prueba. Celda 23. |
| 7 | Codificación de etiquetas y reserva de memoria | Crea `label_to_idx` e `idx_to_label`. `load_images` reserva un arreglo de imágenes `float32` y otro de etiquetas `int64` por partición. Celda 24. |
| 8 | Lectura y conversión de imágenes | Dentro de `load_images`, abre cada archivo con Pillow y convierte sus canales a RGB. El bucle procesa las imágenes una por una y conserva correspondencia entre posición y etiqueta. Celda 24. |
| 9 | Redimensionamiento y normalización | Ajusta cada imagen a 64×64, la convierte a NumPy `float32` y divide entre 255 para obtener valores entre 0 y 1. El tensor tiene forma `[N,64,64,3]`. Celda 24. |
| 10 | Almacenamiento y revisión de datos preparados | Guarda seis arreglos `.npy` y `label_mapping.json` en `data/processed/`; compara visualmente una imagen original con su versión reducida. Celdas 26 y 28. |
| 11 | Construcción de CNN-A | `build_cnn_a` crea dos convoluciones de 32/64 filtros con ReLU y pooling, seguidas de Flatten, Dense de 128 unidades, Dropout de 0.3 y salida softmax de 29 clases. Compila con Adam y pérdida sparse categorical crossentropy. Celda 31. |
| 12 | Entrenamiento de CNN-A | `cnn_a.fit` procesa lotes de entrenamiento, calcula predicciones/pérdida, obtiene gradientes y actualiza pesos con Adam; valida al finalizar cada época. Usa batch 128, máximo 15 épocas y EarlyStopping. Celda 32. |
| 13 | Construcción y entrenamiento de CNN-B | `build_cnn_b` crea cinco convoluciones de 32/32/64/64/128 filtros, tres pooling, BatchNormalization, Dropout, Dense de 256 unidades y softmax de 29 clases. Se entrena con la misma configuración general de fit. Celdas 33–34. |
| 14 | Evaluación y diagnóstico de las CNN | Grafica accuracy/loss; `evaluate` obtiene métricas de prueba y selecciona la mejor CNN por test accuracy. `predict` y `argmax` producen clases para calcular matriz de confusión y reporte de clasificación. Celdas 35, 36 y 38. |
| 15 | Construcción y entrenamiento de redes densas | `build_fc` aplana 64×64×3 a 12,288 valores y aplica BatchNormalization. FC-A usa capas 128/64 y FC-B 512/256/128, con Dropout y softmax de 29 clases; ambas entrenan con Adam y se evalúan en prueba. Celdas 41–43. |
| 16 | Reducción de dimensionalidad mediante PCA | Aplana los tensores y ajusta PCA únicamente con entrenamiento; proyecta train/val/test a 150 componentes. Las salidas históricas conservan aproximadamente 96.2% de varianza explicada. Celda 46. |
| 17 | Entrenamiento y selección de Random Forest | Entrena RF-A con 100 árboles y RF-B con 300/profundidad máxima 30 sobre datos PCA. Selecciona el bosque por accuracy de validación y evalúa el ganador en prueba; ambos solicitan `n_jobs=-1`. Celda 47. |
| 18 | Comparación inicial de modelos | Construye una tabla y gráfica de accuracy en prueba para CNN-A, CNN-B, FC-A, FC-B y el bosque seleccionado. Celda 50. |
| 19 | Aumento de imágenes y reentrenamiento | Incorpora rotación 0.04, zoom 0.1, traslación 0.1/0.1 y contraste 0.1 antes de una nueva instancia de la CNN seleccionada. Entrena desde cero y evalúa el resultado; las transformaciones están dentro del modelo. Celdas 52–53. |
| 20 | Selección del modelo final | Añade la variante aumentada a la comparación y elige el mayor test accuracy. La salida histórica selecciona CNN-B+aug. Celda 56. |
| 21 | Persistencia y recarga de modelos | Guarda cinco redes `.keras`, el bosque, PCA y metadatos `.joblib`. La sección de recarga recupera además dos arreglos test y el mapeo, reevalúa seis modelos y reconstruye la selección sin entrenarlos nuevamente. Celdas 58 y 61. |
| 22 | Recepción de imágenes propias | Ordena los archivos de `imagenes_grupo/` y toma como etiqueta real el texto antes del primer guion bajo del nombre. Hay 15 fotos en el checkout; el código solo exige un mínimo de 5. Celda 62. |
| 23 | Preparación individual para inferencia | `preprocess_single_image` aplica RGB, resize 64×64 y normalización /255, equivalentes a la preparación base del dataset. Se vuelve a invocar al comparar todos los modelos. Celdas 62 y 64. |
| 24 | Predicción y decodificación de la clase | `predict_with_final_model` añade dimensión de batch y obtiene la clase con mayor probabilidad si el modelo es neuronal; si es Random Forest, aplana y proyecta con PCA antes de predecir. Traduce el índice a la etiqueta mediante `idx_to_label`. Celda 62. |
| 25 | Presentación del resultado por imagen | Construye `df_own` con archivo, clase real, predicción y acierto. Muestra cada foto con su etiqueta real/predicha y calcula la proporción de aciertos. Celdas 62–63. |
| 26 | Comparación de generalización y reflexión | `predict_label_generic` evalúa los seis modelos sobre las mismas fotos; resume accuracy interno y externo. El texto final discute distribución, sesgos y limitaciones del prototipo. Celdas 64–67. |

Recorrido mínimo de inferencia después del entrenamiento: archivo de imagen →
RGB → 64×64 → float32/255 → modelo → índice de clase → etiqueta → tabla/visualización.
La rama Random Forest intercala aplanamiento y PCA antes del modelo.

## Parámetros que determinan las repeticiones

Valores iniciales: C=29 clases, S=600 imágenes/clase, N=17,400;
Ntrain=12,180, Nval=Ntest=2,610; B=128. Estos valores describen el laboratorio,
no fijan todavía la configuración del experimento CUDA.

- `E_m`: épocas realmente ejecutadas por cada modelo neuronal m, con máximo 15.
- `M=ceil(Ntrain/B)=96`: lotes por época; 95 lotes de 128 imágenes y uno de 20,
  si se procesa la partición completa sin descartar el último lote.
- `P=15`: fotos propias presentes; `K=6`: entradas del registro final de modelos
  (cinco redes neuronales y un Random Forest seleccionado).
- `b`: tamaño efectivo de un lote; normalmente 128 y 20 en el último de entrenamiento.

Las cantidades se refieren a una pasada por las celdas indicadas, no al número
de veces que una persona haya ejecutado el notebook. Un contador de celda o un
log histórico no permite reconstruir todas las sesiones realizadas.

## Tabla 2. Operaciones que se ejecutan repetidamente

| Operación repetitiva | Datos que procesa | Cantidad de repeticiones |
|---|---|---|
| Inventario por clase | Carpetas y rutas de imágenes | 29 conteos de carpetas en celda 6; el mismo patrón de enumeración vuelve a usarse en EDA y muestreo. No implica decodificar 87,000 imágenes. |
| Inspección visual y de formato | Archivos JPEG y metadatos | Celda 9: 7 clases×5=35 aperturas; celda 12: 10×5=50; celda 18: 6×4=24. Son 109 aperturas de muestra, con posibles archivos repetidos entre actividades. |
| Muestreo y creación de registros | Rutas y nombres de clase | 29 selecciones por clase y 17,400 registros en la submuestra histórica, celda 22. |
| Partición estratificada | DataFrames de rutas/etiquetas | 2 llamadas a `train_test_split`: separar train/resto y luego val/test, celda 23. |
| Reserva de arreglos de entrada/salida | Tensores de imágenes y etiquetas | 3 llamadas a `load_images`, cada una reserva X e y: 6 arreglos, celda 24. |
| Apertura, conversión RGB y resize del dataset | 17,400 imágenes seleccionadas | Una vez por operación e imagen: 12,180+2,610+2,610=17,400; no se repite desde disco en cada época porque fit recibe los arreglos ya cargados. Celda 24. |
| Conversión a float32, normalización y etiquetado | 64×64×3=12,288 valores por imagen y una clase | 17,400 conversiones/normalizaciones y 17,400 asignaciones de etiqueta; se normalizan 213,811,200 valores de canal. Celda 24. |
| Entrenamiento de redes neuronales | X_train/y_train, pesos y estado de Adam | 5 llamadas a fit: CNN-A, CNN-B, FC-A, FC-B y CNN seleccionada+aug. Cada llamada ejecuta E_m épocas de hasta 96 pasos. Celdas 32/34/41/42/52. |
| Forward, pérdida, backward y actualización | Lote de b imágenes, activaciones, etiquetas, pesos y gradientes | Un paso completo por lote: 96×E_m por modelo; máximo 1,440 con 15 épocas. Los cinco historiales suman 6,624 pasos derivados, según la tabla siguiente. |
| Validación y revisión de EarlyStopping | X_val/y_val y val_accuracy | Una pasada de 2,610 imágenes y una revisión del criterio por época: E_m por modelo; 69 pasadas derivadas de los cinco historiales. No actualiza pesos por gradientes. |
| Convolución en forward | Lote, filtros y mapas de activación | Por lote: 2 capas en CNN-A y 5 en CNN-B; la variante aumentada reutiliza la arquitectura elegida. Cada capa produce b×Hout×Wout×Cout valores, cada uno con Kh×Kw×Cin productos. Se aplica también en validación/inferencia, sin backward allí. |
| Pooling en forward | Mapas de activación y ventanas | 2 capas por lote en CNN-A y 3 en CNN-B; cada salida examina su ventana. La variante aumentada hereda el número de la CNN seleccionada. |
| Operaciones densas en forward | Activaciones aplanadas o intermedias y matrices de pesos | Por lote: 2 capas Dense en cada CNN (incluida salida),3 en FC-A y 4 en FC-B; por capa hay b×Nout salidas, cada una reduce Nin productos. |
| BatchNormalization y Dropout | Activaciones por lote/canal o por unidad | Por forward de entrenamiento: CNN-A 0 BN/1 Dropout; CNN-B 4 BN/4 Dropout; FC-A 1 BN/2 Dropout; FC-B 1 BN/3 Dropout. En inferencia no se aplican máscaras aleatorias de Dropout ni se recalculan estadísticas de entrenamiento de BN. |
| Aumento aleatorio durante entrenamiento | Lotes de imágenes de la variante aumentada | 4 transformaciones por lote de entrenamiento: 4×96×E_aug invocaciones de capa; históricamente E_aug=13, es decir 4,992. No es el número de operaciones escalares internas ni se aplica aumento aleatorio al evaluar. Celda 52. |
| Ajuste y proyección PCA | Matrices de 12,288 características por imagen | 1 `fit_transform` sobre train y 2 `transform` sobre val/test, celda 46; otra proyección test durante recarga, celda 61. No se reajusta PCA con test. |
| Entrenamiento de Random Forest | X_train_pca [12180,150] y etiquetas | 2 llamadas a fit; 100+300=400 árboles configurados entre ambos bosques. La cantidad interna de nodos/comparaciones depende del ajuste y no se deduce solo de esos parámetros. Celda 47. |
| Evaluación de Random Forest | Matrices PCA de val/test | 2 predicciones val (una por bosque) y 1 predicción test del ganador, celda 47; otra predicción test al recargar, celda 61. |
| Evaluación neuronal explícita en test | X_test/y_test de 2,610 imágenes | 5 llamadas evaluate iniciales (2 CNN + 2 FC + 1 aumentada) y 5 adicionales al recargar: 10 pasadas si se ejecutan ambas secciones completas. No incluye la validación dentro de fit. |
| Predicción test para diagnóstico de clases | X_test de 2,610 imágenes y 29 probabilidades por imagen | Una llamada predict sobre la mejor CNN, seguida de 2,610 selecciones argmax, matriz de confusión y reporte. Celda 38. |
| Persistencia de artefactos | Arreglos, etiquetas, redes y estimadores | Celda 26: 6 `.npy`+1JSON. Celda 58: 5 `.keras`+3 `.joblib` (bosque, PCA y metadatos). Son cantidades de escrituras de alto nivel, no operaciones físicas de disco. |
| Preprocesamiento de fotos propias | P archivos RGB, reducidos a 64×64 | P=15 invocaciones en celda 62 y otras P=15 en celda 64: 30 en total; en celda 63 se reabren 15 imágenes para mostrarlas sin repetir el preprocesamiento normalizado. |
| Predicción individual del modelo final | Un tensor [1,64,64,3], o vector PCA si es RF | P=15 llamadas predict, una por foto en celda 62; cada resultado se convierte en etiqueta y se compara con su clase real. |
| Predicción de todos los modelos en fotos propias | P=15 imágenes preparadas y K=6 modelos | K×P=90 llamadas predict en celda 64 (75 neuronales y 15 del RF); las imágenes se preparan antes del bucle de modelos. Esta sección vuelve a evaluar el modelo final. |
| Cálculo de accuracy externo y visualización | Booleanos de acierto y fotos | Una media de 15 aciertos/fallos del modelo final en celda 63, 6 medias de 15 comparaciones en celda 64 y 15 paneles de fotos en celda 63. |

Las filas de capas, pasos de entrenamiento y épocas describen niveles anidados
del mismo trabajo: **no se suman entre sí**. Los conteos de invocaciones de capa
no equivalen a lanzamientos de kernels, FLOPs medidos ni número de hilos.

## Repeticiones derivadas de los historiales guardados

| Modelo / celda de fit | Épocas visibles | Pasos derivados (96×épocas) | Presentaciones de imágenes train (12,180×épocas) |
|---|---:|---:|---:|
| CNN-A /32 | 15 | 1,440 | 182,700 |
| CNN-B /34 | 11 | 1,056 | 133,980 |
| FC-A /41 | 15 | 1,440 | 182,700 |
| FC-B /42 | 15 | 1,440 | 182,700 |
| CNN-B+aug /52, selección histórica | 13 | 1,248 | 158,340 |
| Total derivado | 69 | 6,624 | 840,420 |

Estos conteos se obtienen de las etiquetas `Epoch n/15` persistidas y la carga
registrada; no representan una nueva ejecución. Hay 69×2,610=180,090
presentaciones de imágenes de validación si se recorre val completo por época.
Las imágenes se reutilizan entre épocas/modelos: no son 840,420 archivos distintos.

EPOCHS=15 es un máximo. EarlyStopping con paciencia 3 restaura los mejores pesos;
las salidas guardadas registran 15 épocas para CNN-A, 11 para CNN-B y 13 para
CNN-B+aug. No usar duraciones de cargas diferentes como comparación CPU/GPU.

## Antecedentes, no resultados CUDA

| Modelo | Accuracy test histórico |
|---|---:|
| CNN-A | 0.927203 |
| CNN-B | 0.955172 |
| FC-A | 0.530651 |
| FC-B | 0.420307 |
| RF-B | 0.897318 |
| CNN-B+aug | 0.955939 |

CNN-B+aug acertó 3/15 fotos propias. Las imágenes verticales se deforman al
redimensionarse directamente a un cuadrado. El cambio de distribución es una
hipótesis respaldada por diferencias de entrada, no una medición causal aislada.

El set oficial de 28 imágenes se inventaría, pero el código no lo usa para
calcular la métrica principal ni implementa una evaluación adicional sobre él.
La evaluación principal usa el test propio de 2,610 imágenes y las 15 fotos del equipo.

## Límites conocidos

- No hay configuración CUDA, temporizador experimental ni registro de transferencias.
- La carga de imágenes no implementa explícitamente preparación con varios núcleos.
- No se fija semilla de TensorFlow; no se ordena el listado antes de muestrear.
- Se selecciona CNN y modelo final usando test: separar selección por validación
  y evaluación final, o documentar el sesgo heredado si se conserva.
- Las salidas incluyen rutas de distintas máquinas y contadores reiniciados.
  No prueban una única ejecución de principio a fin.
- Los arreglos/modelos persistidos no están presentes en este checkout.
- Los README originales son mínimos y requirements.txt no fija versiones.
- Las convoluciones y densas son candidatos a carga dominante. Falta medir para
  afirmar cuáles son los cuellos de botella y qué porcentaje representan.

Memoria derivada de dimensiones: N×64×64×3×4 = 855,244,800 bytes solo en imágenes
float32. No incluye copias, etiquetas, activaciones ni optimizador.
