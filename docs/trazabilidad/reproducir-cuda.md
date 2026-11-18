# Reproducir o consultar la implementación CUDA sin repetir trabajo

REQ-06/07/12; CAM-006. Referencia:EXP-003. La guía distingue consulta de evidencia,
regeneración de informe y **otra medición**, que siempre requiere identidad nueva.
No se ejecutan estos comandos automáticamente al leer esta guía.

## Consultar lo ya ejecutado

Abrir [notebook fuente](../../notebooks/Lab3_ASL_CUDA.ipynb). La CPU original está
desactivada. ASL_CUDA_RUN=1 habilita el controlador; si EXP-003 ya está completo,
run_series devuelve su registro y no entrena otra vez. Para revisar código no
es necesario habilitar mediciones ni importar TensorFlow.

La copia con salidas reales y el informe están en resultados/paralelo/EXP-003.
El código de entrenamiento,CSV,logs y modelos referenciados en data/ son las
evidencias. Los modelos son archivos grandes excluidos de Git; no se reconstruyen
ni se inventa su contenido si faltan. Los hashes permiten detectar esa ausencia.

## Requisitos de ejecución

1. Windowsx64,PCUDA Python3.14.3 y [paquetes fijados](../../resultados/preparacion/cuda/requirements-pcuda.txt).
   [Registro de entorno](../../resultados/preparacion/cuda/entorno.md). No reinstalar
   dependencias ya presentes. La reconstrucción limpia del entorno completo no
   está certificada; requisitos fijados no garantizan disponibilidad perpetua de
   paquetes nightly. Conservar instaladores/ruedas si se migra de máquina.
2. GPU NVIDIA compatible sm_75,driver591.59/toolkit13.1 del ensayo o un entorno
   nuevo documentado y validado. La implementación local usa DLLWindows cuBLAS13
   de CUDA_HOME/bin/x64. No es una distribución Linux validada.
3. Manifiestos,label_mapping y estado inicial preservados de EXP-002. Archivo
   inicial en data/EXP-002/RUN-20261117T171942Z-cpu-01/CNN-B.estado-inicial.npz.
4. Dataset local con rutas relativas del manifiesto. No se descarga otra vez:
   restaurar desde la [copia seleccionada preservada](../../resultados/secuencial/EXP-002/preservacion-datos.md)
   si no está la caché. Conservar también el NPZ inicial; no sustituirlo por una
   nueva inicialización con seed43.
5. Memoria disponible de admisión:al menos4GiBHost;al menos1536MiBVRAM tras
   contexto y antes de buffers;tope1408MiB de buffers propios. Sin otra carga
   experimental simultánea y con plan de energía/condiciones registrados.

Todos los RUN verifican fuente/datos antes de medir y X/y preparados por SHA-256
contra CPU. Se usa la misma selección y split, sin muestreo nuevo ni test usado
como entrenamiento. Los cuatro RNGDropout originales se conservan en el NPZ
como procedencia; sus máscaras históricas no son requisito para medir la carga.

## Otra medición, solo si se desea expresamente

El [preparador](../../scripts/preparar_reproduccion_cuda.py) admite una raíz de
datos local diferente sin editar el experimento congelado. Rechaza destinos
existentes,fuentes/paquetes distintos y estado inicial incorrecto. Ejemplo
**pendiente de ejecutar**, con un ID que no exista:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/preparar_reproduccion_cuda.py --experimento EXP-004 --dataset "C:\Users\David\.cache\kagglehub\datasets\grassknoted\asl-alphabet\versions\1"
.\PCUDA\Scripts\python.exe -u -X utf8 scripts/benchmark_cuda.py run --experimento EXP-004
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cuda.py --experimento EXP-004
```

La captura de pantalla del informe actual pertenece a EXP-003 y no se reutiliza
como prueba de ejecución del experimento nuevo. Un nuevo EXP debe generar sus
propias capturas,contexto y fuentes. El informe debe identificarlo y no mezclar
sus observaciones con las anteriores ni atribuir fotos viejas a hardware nuevo.

El notebook entregado consulta EXP-003; la CLI permite otra identidad sin editar
el notebook fuente congelado. Una serie incompleta/fallida se conserva y requiere
diagnóstico,no sobrescritura. No es necesario volver a ejecutar la campaña CPU
si se mantiene la referencia compatible; un cambio real de carga/plataforma se
declara y exige valorar la comparación antes de presentar un speedup.

## Pruebas y límites

Las pruebas pequeñas de operadores,adjuntos e integración tienen directorios
inmutables bajo resultados/preparacion/cuda. Se pueden repetir con otro destino;
no son nuevas corridas CPU completas. Los casos de red completa que fallaron
por sensibilidad de rutas MaxPooling se conservan e interpretan medianteDEC-023.
No se promete identidad numérica entre GPU y CPU ni entre plataformas distintas.

La identidad del archivo .keras incluye metadatos de contenedor; para comparar
estados numéricos se utiliza además el hash canónico de datasets HDF5. Contador
Adam1440,15épocas y12180muestras por época se verifican independientemente de
la duración. El tiempo de pared incluye transferencias y validaciónCPU, y los
eventos se reportan como intervalos de stream, no como tiempo activo puro.
