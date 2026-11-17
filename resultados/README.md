# Resultados y evidencias

Estado inicial: **sin mediciones experimentales CPU/GPU**. Los CSV solo tienen
cabeceras; no contienen resultados cero ni observaciones ficticias.

| Carpeta / archivo | Uso |
|---|---|
| [inventario-inicial.csv](inventario-inicial.csv) | Identidad de la consigna, notebook, dependencias y fotos al preparar trazabilidad |
| [preparacion](preparacion/README.md) | Comprobaciones documentales y de herramientas; no son benchmarks |
| [secuencial](secuencial/README.md) | Resultados detallados CPU y corridas originales |
| [paralelo](paralelo/README.md) | Resultados detallados CUDA, transferencias y recursos |
| [comparacion](comparacion/README.md) | Emparejamiento, estadísticas, speedup y figuras |

## Estructura de una corrida

```text
secuencial/ o paralelo/
  RUN-AAAAMMDDTHHMMSSZ-cpu-o-gpu-NN/
    registro.md
    configuracion.json
    contexto/                 # captura antes de medir
    manifiestos/              # datos, splits, mapeo o referencias con hashes
    tiempos.csv
    logs/                     # comando, stdout, stderr, monitorización
    capturas/                 # originales, fase e intervalo identificados
    validacion/               # corrección y métricas
    artefactos.csv            # manifiesto final de la corrida
```

Cada carpeta conserva una ejecución y referencia su EXP. Mantener todos los
valores individuales y fallos. El índice general `tiempos.csv` de CPU/GPU puede
consolidarlos, pero debe conservar RUN y fase para rastrear cada fila al original.

## Esquema tiempos.csv

| Columna | Significado |
|---|---|
| experimento_id,run_id | Vínculos con EXP y registro.md |
| dispositivo | cpu o gpu |
| repeticion | Índice de repetición medido |
| fase | Nombre definido en reproducibilidad.md |
| tipo_medicion | pared, evento_device o profiler; no mezclar al promediar |
| duracion_s | Segundos decimales, sin unidades en la celda |
| inicio_utc,fin_utc | Intervalo temporal de la fase |
| epocas,pasos,muestras,batch_size | Carga realmente ejecutada en el intervalo |
| incluido_resumen | true/false; warmups y fallos se excluyen justificadamente |
| estado | completada/fallida/interrumpida/no_aplica/pendiente |
| configuracion_sha256,datos_sha256 | Identidad de configuración común y manifiesto de datos |
| evidencia | Ruta relativa al archivo que respalda la observación |
| motivo | Exclusión, no aplicabilidad, error u observación |

Una duración desconocida queda vacía con estado pendiente. Para CPU, h2d/d2h
tienen estado no_aplica y duración vacía. Los intervalos se documentan en el reloj
de pared; los eventos de dispositivo conservan además su salida cruda.

No existe todavía un exportador de métricas del algoritmo: la implementación
CPU/CUDA deberá producir este esquema o documentar su conversión verificable.

## Manifiesto inicial

Se generó mediante lectura de bytes y SHA-256, sin ejecutar el notebook. Los
paths son relativos a la raíz. Es una foto de entradas iniciales, no se regenera
silenciosamente si el original cambia; se crea una nueva versión y un CAM.
