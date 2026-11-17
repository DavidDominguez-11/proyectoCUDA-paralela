# Resultados detallados paralelos / CUDA

Estado: pendiente diseñar, implementar y ejecutar PROC-07/08.

Por cada EXP publicar un `resultados.md` con:

1. Modelo/operaciones y enlace al diseño de threads, blocks, grids y memoria.
2. Datos, configuración común y referencia CPU compatibles, con hashes.
3. GPU/VRAM, driver, CUDA/cuDNN efectivos, API y evidencia de uso real.
4. Repeticiones, warmup, restauración de estado, semillas y sincronización.
5. H2D, cómputo, D2H y total por repetición; transferencias incluidas y frecuencia.
6. Coste de inicialización/compilación, memoria asignada/pico y solapamiento.
7. Estadísticas por fase, resultados de corrección/calidad y variación frente a CPU.
8. Logs y capturas durante entrenamiento; evidencia de ocupación/uso de cómputo
   y VRAM disponible. No llamar ocupación de kernels a un simple porcentaje de uso.
9. Fallos, límites de tamaño, sobrecostes y datos fuente de gráficas.

Usar [la plantilla de ejecución](../../docs/trazabilidad/plantillas/ejecucion.md).
El CSV consolidado comienza vacío. Un mensaje de dispositivo detectado no
demuestra ejecución de la matemática en GPU.
