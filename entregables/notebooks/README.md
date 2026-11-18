# Código y salidas completas, por separado

- [ASL_Secuencial.ipynb](ASL_Secuencial.ipynb):37celdas de código ejecutadas,
  75épocas nuevas(cinco redes),PCA,bosques,inferencia,10imágenes y0errores.
- [ASL_CUDA.ipynb](ASL_CUDA.ipynb):31celdas de código ejecutadas,
  15épocas nuevasCNN-B,1440actualizaciones,4imágenes y0errores.

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
| ASL_CUDA.ipynb | 89c14f3184c36f7b63a53016dc20427c6b131ccc9196849d5e36ee0163e39595 |
