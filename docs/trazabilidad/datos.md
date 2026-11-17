# Datos, procedencia y evidencia

## Registro de datos: estado tras el paso 2

| ID | Conjunto | Procedencia | Versión / identidad | Estado |
|---|---|---|---|---|
| DAT-01 | ASL Alphabet | Caché local grassknoted/asl-alphabet | Versión1 local; 87,000 archivos inventariados y 17,400 usados con SHA-256 | Verificado sin descargar; selección preservada en ZIP local |
| DAT-02 | Fotos del equipo | imagenes_grupo, 15 JPEG de luis/bran/david | Hash individual en inventario-inicial.csv | Archivos presentes; metadatos inspeccionados |
| DAT-03 | Submuestra y splits | Derivados de DAT-01 | DAT-03-EXP-002: 600/clase, semilla42, 70/15/15; manifiesto ordenado | Congelado y utilizado en cinco corridas |
| DAT-04 | Tensores normalizados | Derivados de DAT-03 | RGB,64×64,float32,/255; y int64; hashes por partición | Coinciden entre cinco corridas; reconstruibles desde datos/manifiestos |
| DAT-05 | Estados iniciales completos | Constructores originales y seeds explícitas por arquitectura | NPZ, SHA de archivo y fingerprint de valores en cada RUN | Verificados e iguales entre corridas; transferencia a GPU aún no validada |

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

## Validación para el paso 2 (2026-11-17)

Se confirmó la caché local indicada por el usuario sin descargar:29 clases y
87,000 imágenes de entrenamiento disponibles, más28 de prueba oficial. Se
seleccionaron17,400 imágenes con las funciones originales y se congelaron
train/val/test de12,180/2,610/2,610. Cada archivo seleccionado se leyó para SHA-256
y se verificó con Pillow. No hubo grupos de contenido idéntico en la selección;
esto no demuestra independencia de sujetos o sesiones de captura.

- DAT-01-v1: fuente local validada; no se redistribuyen imágenes del dataset.
- DAT-03-EXP-002: [manifiesto de particiones](../../resultados/secuencial/EXP-002/manifest_splits.csv).
- [Manifiesto de fuentes](../../resultados/secuencial/EXP-002/manifest_fuente.csv) y
  [auditoría](../../resultados/secuencial/EXP-002/validacion_datos.json).
- DAT-04: cada RUN guarda identidad de tensores preparados en tensores.json;
  se reconstruyen desde archivos y manifiesto, sin persistir copias .npy redundantes.
- DAT-05: cada RUN conserva estados completos iniciales por arquitectura y sus
  fingerprints en estados-iniciales.json. Binarios en data/EXP-002, con hashes.
- EXP-001 anterior se conserva como serie interrumpida por el monitor incorrecto;
  no aporta observaciones al promedio. EXP-002 vuelve a congelar la misma carga
  con identificación nueva y monitor corregido.

El informe final CPU contrastó los fingerprints: coinciden entre las cinco corridas.
Los checkpoints están disponibles localmente y pueden reconstruirse con el
entorno/semillas registrados; no se ha comprobado aún su recuperación en otra máquina.

Se preservaron las 17,400 imágenes en un ZIP local determinista de 224,125,103 bytes:
[registro de preservación](../../resultados/secuencial/EXP-002/preservacion-datos.md).
SHA-256: `6af5baba0a6049b489263f5b6e159375f1e281ff4c23b61a950d91f557e1c23d`.
La copia se creó después de cronometrar; no intervino en los tiempos medidos.

## Política de conservación

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
