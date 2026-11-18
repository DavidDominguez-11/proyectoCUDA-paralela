# Resultados y evidencias

Estado actual: **cinco corridas CPU EXP-002 y cinco CUDA EXP-003 completadas**.
El [informe final DOCX](../entregables/Informe_Proyecto_3_CUDA.docx) integra los
seis pasos y evidencias;[revisión y cobertura](../docs/trazabilidad/informe-final.md).
No se exportó un PDF final:el usuario realizará ese paso.
Los valores crudos se conservan por RUN y el índice CPU está consolidado.
El [informe CPU](secuencial/EXP-002/resultados.md) separa las fases comunes de
las ramas adaptativas; no se mezclan como si fueran una única carga fija.

El [informe CUDA CNN-B](paralelo/EXP-003/resultados.md) reutiliza la fase CPU
compatible y registra transferencias,repeticiones,calidad y límites. Hay
[PDF](paralelo/EXP-003/informe-paso-5.pdf) y
[capturas reales](evidencias/EXP-003/README.md).

El [paso 6: análisis y conclusiones](comparacion/paso-6-revision-02/analisis-y-conclusiones.md)
compara medias/variabilidad por fase,calcula speedup/ahorro y discute causas y
límites. Es Markdown con CSV y tres gráficas PNG/SVG;no es informe final ni PDF.

| Carpeta / archivo | Uso |
|---|---|
| [inventario-inicial.csv](inventario-inicial.csv) | Identidad de la consigna, notebook, dependencias y fotos al preparar trazabilidad |
| [analisis-original/revision-estatica.md](analisis-original/revision-estatica.md) | Evidencia del paso 1: código y salidas históricas extraídos sin ejecutar el modelo |
| [diseno/paso-4-inspeccion.md](diseno/paso-4-inspeccion.md) | Inventario y cálculos del diseño; sin kernels ni benchmark GPU |
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

benchmark_cpu.py y benchmark_cuda.py producen este esquema. En la serie CUDA,
dispositivo identifica la variante; tiempos_clasificados.csv añade el ejecutor
efectivo de cada fase para distinguir CPU,GPU,transferencias y tramo heterogéneo.

## Manifiesto inicial

Se generó mediante lectura de bytes y SHA-256, sin ejecutar el notebook. Los
paths son relativos a la raíz. Es una foto de entradas iniciales, no se regenera
silenciosamente si el original cambia; se crea una nueva versión y un CAM.
