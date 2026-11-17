# Datos, procedencia y evidencia

## Registro inicial

| ID | Conjunto | Procedencia | Versión / identidad | Estado |
|---|---|---|---|---|
| DAT-01 | ASL Alphabet | grassknoted/asl-alphabet, descargado con kagglehub | Salida histórica: versions/1; identidad real del experimento pendiente | Fuera del repositorio; no validado ahora |
| DAT-02 | Fotos del equipo | imagenes_grupo, 15 JPEG de luis/bran/david | Hash individual en inventario-inicial.csv | Archivos presentes; metadatos inspeccionados |
| DAT-03 | Submuestra y splits | Derivados de DAT-01 | Histórico: 600/clase, semilla42, 70/15/15 | Manifiesto reproducible pendiente |
| DAT-04 | Tensores normalizados | Derivados de DAT-03 | RGB,64×64,float32,/255; y int64 | Arreglos locales no presentes |
| DAT-05 | Pesos iniciales de referencia | Modelo y semilla a decidir | Archivo y SHA-256 pendientes | Deben compartirse entre CPU y GPU |

El enlace del dataset se conserva como procedencia histórica:
https://www.kaggle.com/datasets/grassknoted/asl-alphabet.
No se consultó ni descargó para esta preparación. Registrar licencia y condiciones
de la versión realmente adquirida antes de redistribuir archivos.

## Identidad y manifiestos obligatorios por experimento

1. Registrar proveedor, identificador, versión exacta, fecha UTC de adquisición,
   comando, versión de kagglehub, ruta local y tamaño. Evitar `latest` sin registrar
   la versión resuelta. Una ruta de caché no sustituye al hash de los archivos.
2. Crear `manifest_fuente.csv` con una fila por archivo usado. Columnas:
   `dataset_id,ruta_relativa,bytes,sha256,clase,ancho,alto,modo,estado,motivo`.
3. Crear `manifest_splits.csv` con
   `dataset_id,ruta_relativa,sha256,clase,label_idx,split,orden`.
   Debe ser suficiente para reconstruir la selección sin volver a sortear archivos.
4. Conservar `label_mapping.json`, parámetros de transformación y hashes de los
   arreglos derivados. Registrar filtro de resize, política de orientación EXIF,
   conversión de color, rango y dtype. No dejar estos comportamientos implícitos.
5. Verificar conteos por split/clase, ausencia de intersección de rutas y hashes,
   archivos corruptos y posibles duplicados entre particiones. Registrar acciones,
   exclusiones y efectos sobre N. No afirmar independencia de sujetos/sesiones
   cuando el dataset no la demuestra.
6. Cerrar el manifiesto y registrar su SHA-256 en la configuración de CPU y GPU.
   Si cambia una entrada, abrir una nueva versión DAT y revisar comparabilidad.

## Registro de transformación

| Paso | Entrada → salida | Parámetros a conservar | Evidencia |
|---|---|---|---|
| Enumeración | Directorios → archivos | Orden estable, extensiones y exclusiones | manifest_fuente.csv |
| Muestreo | Fuente → selección | Algoritmo, semilla y máximo por clase | manifest_splits.csv |
| División | Selección → train/val/test | Estratificación, proporciones, semilla | Conteos y comprobación de separación |
| Preparación | JPEG → tensor | RGB, tamaño, resize, EXIF, float32, /255 | Configuración, muestra visual, hashes |
| Etiquetado | Clase → entero | Orden exacto de las 29 clases | label_mapping.json |
| Aumento, si aplica | Tensor → muestra transformada | Ubicación CPU, semilla, parámetros, frecuencia | DEC y configuración |
| Pesos iniciales | Modelo → checkpoint inicial | Arquitectura, dtype, semilla y estado optimizador | Archivo o ubicación recuperable y SHA-256 |

## Conservación de datos grandes

Los datos crudos, .npy y modelos pueden mantenerse fuera de Git; sus manifiestos,
metadatos y procedimientos de recuperación deben estar versionados. Registrar
ubicación durable, responsable, versión, tamaño, SHA-256 y comando de recuperación
de todo archivo externo. Una caché personal o una URL expirable no basta.

Mantener manifiestos, configuraciones, resúmenes, tiempos CSV y gráficas finales en
el repositorio. Si se externalizan logs o capturas por tamaño, dejar un índice
Markdown con ubicación recuperable y hashes. No eliminar los originales.

Las fotos del equipo muestran resoluciones 1200×1600 (luis/bran) y 720×1280
(david), en RGB. Los hashes prueban identidad, no corrección de etiquetas ni
representatividad estadística; esas validaciones deben documentarse por separado.
