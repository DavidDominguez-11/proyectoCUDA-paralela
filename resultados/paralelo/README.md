# Resultados detallados paralelos / CUDA

Estado: **CNN-B implementada y ejecutada; cinco corridas reales EXP-003**.

[Informe detallado](EXP-003/resultados.md), [PDF](EXP-003/informe-paso-5.pdf),
[notebook ejecutado](EXP-003/notebook-validado.ipynb) y
[capturas](../evidencias/EXP-003/README.md). Media entrenamiento418.608s,
incluyendo transferencias y validaciónCPU; speedup1.925× respecto a CPU histórica
compatible, sin repetir su campaña. El alcance es una CNN-B, no todo el notebook.

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
El CSV consolidado contiene las observaciones reales de los cinco RUN. Un mensaje de dispositivo detectado no
demuestra ejecución de la matemática en GPU.
