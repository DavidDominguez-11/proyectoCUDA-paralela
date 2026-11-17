# Resultados detallados secuenciales / CPU

Estado: PROC-06 completado con cinco corridas CPU en PCUDA, sin reinstalación ni
descarga. [Informe completo con tablas y gráficas](EXP-002/resultados.md).

| Medición | Media | Desviación muestral | Interpretación |
|---|---:|---:|---|
| Referencia común, suma de 14 fases | 1119.928 s | 2.723 s | Misma carga base en cinco corridas; IC nominal 95%: 1116.546–1123.309 s |
| Programa adaptativo completo | 1858.617 s | 314.366 s | Mezcla de ramas; dato descriptivo, no baseline homogéneo GPU |

La cuarta corrida seleccionó CNN-A para el aumento; las otras cuatro CNN-B. No
se excluyó la corrida rápida. La revisión de alcance está en DEC-015 y se hizo
antes del resultado de la quinta. No se declara cumplida la regla inicial de
precisión del total mezclado. La referencia común tiene CV de 0.24%.

- [Configuración congelada](../../configuraciones/EXP-002.json).
- [Estado de la serie](EXP-002/serie.json).
- [Manifiesto de datos](EXP-002/manifest_fuente.csv).
- [Particiones exactas](EXP-002/manifest_splits.csv).
- [Resumen de fases estratificadas](EXP-002/resumen.csv).
- [Referencia común por repetición](EXP-002/tiempos_referencia_comun.csv).
- [Reproducción concreta](../../docs/trazabilidad/reproducir-exp-002.md).

EXP-001 se conserva como intento interrumpido por un monitor que observaba el
lanzador del venv en lugar de TensorFlow. Sus tiempos parciales no se incorporan
al promedio. EXP-002 monitoriza el árbol completo de procesos; ver BIT-005.
La carpeta de extensión CPU-06 solo contiene preparación de contexto cancelada;
no es una sexta observación.

Por cada EXP publicar un `resultados.md` al finalizar, con:

1. Objetivo, configuración y datos/hashes; alcance exacto de la referencia CPU.
2. Hardware, entorno, hilos matemáticos, workers y evidencia de dispositivo efectivo.
3. Procedimiento, repeticiones, warmup, semillas y estado inicial del modelo.
4. Tiempos individuales por fase, unidades, RUN y valores excluidos con su causa.
5. Promedio, mediana, desviación muestral, mínimo y máximo por fase comparable.
6. Gráficas con CSV/script fuente, parámetros y versiones para regenerarlas.
7. Capturas de preparación por núcleo y logs correlacionados con fases.
8. Calidad/corrección, errores, cuellos observados y limitaciones.

Usar [la plantilla de ejecución](../../docs/trazabilidad/plantillas/ejecucion.md)
y conservar `tiempos.csv` por RUN. El CSV de esta carpeta consolida las 125 filas
originales de las cinco corridas; incluye subintervalos anidados identificados.

Las duraciones de épocas guardadas en el notebook son antecedentes de distintas
sesiones; no equivalen a este benchmark repetido ni a una referencia monohilo.
