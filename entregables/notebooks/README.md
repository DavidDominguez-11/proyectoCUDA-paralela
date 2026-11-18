# Código y salidas completas, por separado

- [ASL_Secuencial.ipynb](ASL_Secuencial.ipynb):37celdas de código ejecutadas,
  75épocas nuevas(cinco redes),PCA,bosques,inferencia,10imágenes y0errores.
- [ASL_CUDA.ipynb](ASL_CUDA.ipynb):92celdas de código de sesiones nativas identificadas, CNN-B más cinco corridas FC-A y cinco FC-B;30imágenes y0errores. Conserva las53celdas anteriores, con150épocas FC añadidas.

Ambos contienen las salidas nativas junto al código,datos/condiciones,tablas,
métricas,curvas,logs y capturas nuevas. Incluyen además los registros íntegros
de las cinco mediciones históricas por variante,identificados aparte. Las
imágenes/salidas están incrustadas:se pueden abrir sin volver a entrenar.

La versiónCPU contiene el programa secuencial;la CUDA separa preparaciónCPU,
entrenamientoGPU y validación/testCPU,con los módulosCUDA completos ejecutados
en sus celdas. El código no simula el entrenamiento leyendo un historial.

Corridas adicionales:EXP-008CPU yEXP-009CUDA. No sustituyen las cinco mediciones
del informe. El total continuo de notebook incluye presentación entre fases;
fit/evaluate conservan sus fronteras. Para calcular otra vez se deben preparar
destinos nuevos conPCUDA y el proyecto/dataset presentes;no sobrescribir RUNcerrados.

Los originales ejecutados se preservaron antes de añadir un anexoMarkdown con
los diagnósticos nativos de TensorFlow/kernel. No se cambió código,contador ni
salida de las celdas. [Validación](../../resultados/preparacion/notebooks-entrega-v03/validacion-final.json).

| Archivo | SHA-256 de entrega |
|---|---|
| ASL_Secuencial.ipynb | 201e6dff41312e2b921349ba5c066ca3abd9d25dcae7fb7206a2fd5d29646692 |
| ASL_CUDA.ipynb | 4e9c6b7787cf3dd4ebe3113006d5f1e21350745ab0ea54464b30e067b813a897 |


La ampliación FC reutiliza la primera FC-A ya guardada y ejecuta solo nueve pendientes. No se reentrenó CPU/CNN-B. Los contadores reinician en cada sesión, explícitamente identificada; el cierre de presentación recuperado lee archivos reales sin repetir cálculos. [Procedimiento y validación actuales](../../docs/trazabilidad/ampliacion-fc.md).
