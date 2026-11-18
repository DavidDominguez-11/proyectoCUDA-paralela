# Comparación CPU / GPU

Estado: **comparación CNN-B disponible, cinco referencias CPU EXP-002 y cinco CUDA EXP-003**.

[Resultados y límites](../paralelo/EXP-003/resultados.md),
[comparabilidad](../paralelo/EXP-003/comparabilidad.csv),
[speedup](../paralelo/EXP-003/speedup.csv). Se compara tiempo de pared completo
de entrenamiento, incluyendo transferencias/validaciónCPU;1.924594×.
La referencia es histórica no alternada y los RNG internos difieren; no se afirma
identidad numérica CPU/GPU. La mejora de preparación1→6workers es CPU, no CUDA.

Crear por EXP un `analisis.md`, los CSV derivados y el script/configuración que
los regenera. Registrar su commit, versiones, comandos y hashes de entradas.

## Verificación previa

Emparejar configuraciones comunes, datos, pesos iniciales, precisión, carga,
semillas y alcance temporal. Conservar en `comparabilidad.csv` las columnas
`experimento_id,run_cpu,run_gpu,criterio,valor_cpu,valor_gpu,equivalente,justificacion`.
No promediar mediciones de configuraciones distintas. Una diferencia justificada
de plataforma no debe ocultarse: explicar su influencia y límites de atribución.

## Resultados derivados

- `resumen.csv`: estadísticas de observaciones válidas de CPU y GPU por EXP/fase.
- `speedup.csv`: cociente de medias para fases con intervalos equivalentes.
- Gráficas: tiempos CPU/GPU por fase; dispersión de repeticiones; desglose
  transferencias/cómputo; speedup por fase y pipeline total.
- Preservar unidades, n, definición de barras de error y datos de cada figura.
  No apilar intervalos solapados como si fueran componentes disjuntos.
- Explicar si se redujo el tiempo, las causas apoyadas por evidencia y qué no se
  pudo medir. No convertir hipótesis sobre ocupación o ancho de banda en hechos.

`speedup = media_cpu_s / media_gpu_s`, con transferencia incluida en el total.
Para h2d/d2h se reporta el coste GPU, sin inventar un speedup de transferencia.
La comparación de accuracy y la de velocidad responden preguntas diferentes;
documentar ambas y sus tolerancias.
