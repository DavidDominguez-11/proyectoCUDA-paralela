# Análisis del programa original

Tipo de evidencia: inspección estática del
[notebook original](../../LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb) y lectura
de sus salidas históricas. Se numeran las celdas desde cero; el número es una
referencia inicial, pues puede cambiar al editar. La identidad exacta del archivo
está en [el manifiesto inicial](../../resultados/inventario-inicial.csv).

## Descomposición

| Operación | Celdas / función | Funcionamiento y salida | Ubicación observada en código |
|---|---|---|---|
| Configuración | 2 | Imports, SEED=42, semillas Python/NumPy | Host |
| Adquisición | 4; kagglehub.dataset_download | Descarga o reutiliza caché de ASL Alphabet | Host / archivos |
| Inventario y EDA | 6–20 | Conteos, muestras, tamaños y clases confundibles | Host; visualizaciones |
| Muestreo | 22; build_sampled_file_list | Selecciona hasta 600 archivos por clase | Host |
| Partición | 23; train_test_split | División estratificada 70/15/15 | Host |
| Preparación | 24; load_images | JPEG → RGB → 64×64 → float32 /255; etiquetas int64 | Bucle Host; sin pool explícito |
| Persistencia de datos | 26 | .npy y label_mapping.json | Host / disco |
| CNN-A | 31–32; build_cnn_a | Conv32, pool, Conv64, pool, Flatten, Dense128, Dropout, salida29 | Keras; dispositivo no forzado |
| CNN-B | 33–34; build_cnn_b | Cinco convoluciones 32/32/64/64/128, tres pools, BN, Dropout, Dense256, salida29 | Keras; dispositivo no forzado |
| Evaluación CNN | 35–39 | Curvas, accuracy/loss, predicciones y matriz de confusión | Keras + NumPy/sklearn |
| Redes densas | 41–43; build_fc | Flatten, BN; FC-A 128/64 y FC-B 512/256/128 | Keras; dispositivo no forzado |
| PCA | 46 | 12,288 variables → 150 componentes; ajuste solo sobre train | sklearn / Host |
| Random Forest | 47 | Configuraciones 100 árboles y 300/profundidad30; n_jobs=-1 | CPU con paralelismo solicitado |
| Aumento | 52 | Rotación, zoom, traslación y contraste dentro del modelo; reentrenamiento | Keras; dispositivo no forzado |
| Selección y guardado | 50, 56, 58 | Comparación, selección y persistencia .keras/.joblib | Host + evaluación Keras |
| Recarga e imágenes propias | 61–64 | Recarga modelos/arreglos y clasifica 15 fotos | Host + inferencia |

## Operaciones repetitivas

Valores iniciales: C=29 clases, S=600 imágenes/clase, N=17,400;
Ntrain=12,180, Nval=Ntest=2,610; B=128. Estos valores describen el laboratorio,
no fijan todavía la configuración del experimento CUDA.

| Operación | Datos | Cantidad / fórmula | Dependencias |
|---|---|---|---|
| Decodificación, resize y normalización | Imagen RGB | N=17,400 imágenes por preparación completa | Independiente por archivo; preservar índice y etiqueta |
| Batches de entrenamiento | Tensores [B,64,64,3] | ceil(12180/128)=96 por época | Actualización de pesos entre batches |
| Forward/backward/Adam | Activaciones, pesos y etiquetas | E×96 pasos por entrenamiento si no se descarta el último batch | Paralelismo dentro del paso; épocas no independientes |
| Validación durante fit | 2,610 imágenes | Una evaluación por época ejecutada | Pesos de esa época |
| Convolución | Imagen, filtro, canales | Por capa: B×Hout×Wout×Cout salidas; cada salida reduce Kh×Kw×Cin productos | Reutilización de entrada/filtros y reducción |
| Capa densa | Matrices de activaciones/pesos | Por capa: B×Nout salidas; cada salida reduce Nin productos | Reducción por salida |
| BatchNormalization | Activaciones por canal | Cada capa y batch; reducción según ejes | Sincronización y estadísticas móviles |
| Aumento de imágenes | Batch de imágenes | Por batch de entrenamiento de la variante aumentada | Aleatoriedad y ubicación CPU por resolver |
| Inferencia final propia | 15 fotos | Una predicción por foto y modelo evaluado | Mismo preprocesamiento y mapeo de clases |

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
