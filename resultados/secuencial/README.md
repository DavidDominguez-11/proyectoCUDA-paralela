# Resultados detallados secuenciales / CPU

Estado: pendiente ejecutar PROC-06. No hay tiempos experimentales nuevos.

Por cada EXP publicar un `resultados.md` con:

1. Objetivo, configuración y datos/hashes; alcance exacto de la referencia CPU.
2. Hardware, entorno, hilos matemáticos, workers y evidencia de dispositivo efectivo.
3. Procedimiento, repeticiones, warmup, semillas y estado inicial del modelo.
4. Tiempos individuales por fase, unidades, RUN y valores excluidos con su causa.
5. Promedio, mediana, desviación muestral, mínimo y máximo por fase comparable.
6. Gráficas con CSV/script fuente, parámetros y versiones para regenerarlas.
7. Capturas de preparación por núcleo y logs correlacionados con fases.
8. Calidad/corrección, errores, cuellos observados y limitaciones.

Usar [la plantilla de ejecución](../../docs/trazabilidad/plantillas/ejecucion.md)
y conservar `tiempos.csv` por RUN. El CSV de esta carpeta es un índice consolidado
con cabeceras iniciales, sin datos inventados.

Las duraciones de épocas guardadas en el notebook son antecedentes de distintas
sesiones; no equivalen a este benchmark repetido ni a una referencia monohilo.
