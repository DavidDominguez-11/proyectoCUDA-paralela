# Capturas reales de recursos — EXP-003

Se capturó en pantalla una ventana real de monitorización del experimento,
alimentada por las muestras originales de psutil/nvidia-smi en recursos.csv.
No son gráficas inventadas ni capturas del sistema en reposo. La ventana muestra
UTC,RUN,fase,actividad de12procesadores lógicos,árbol del proceso,GPU y VRAM.

| Evidencia seleccionada | Observación visible |
|---|---|
| [CPU preparación](20261117T233947Z-cpu.png) | RUN-20261117T233923Z-gpu-04,preparaciónCPU;CPU proceso225.2%,GPU1%;seis workers configurados |
| [GPU entrenamiento](20261117T233706Z-gpu.png) | RUN-20261117T233146Z-gpu-03,época11;GPU88%,VRAM1213MiB;CPU proceso98.9% |

Los JSON homónimos conservan muestra original,instante de captura,RUN,ruta CSV
y SHA-256 del PNG:[CPU](20261117T233947Z-cpu.json),[GPU](20261117T233706Z-gpu.json).
La muestra se coteja por su UTC con el CSV del RUN sellado. La captura no demuestra
ocupación de cada SM ni que toda la VRAM esté saturada. CPU proceso puede superar
100% al sumar actividad de varios hilos; la espera/orquestación también usa CPU.

Las imágenes se inspeccionaron visualmente:etiquetas/valores/fase legibles.
El recorte GPU incluye el contorno de la ventana; no se editó después de capturarlo.

## Intento preliminar conservado y excluido

20261117T232956Z-gpu.png y20261117T233210Z-cpu.png se conservan como primer intento.
El escalado DPI produjo recorte incompleto en GPU y una imagen CPU no utilizable;
no se usan como evidencia seleccionada. Se corrigió DPI en el capturador antes de
repetir con nuevos nombres, sin sobrescribir archivos ni alterar cargas o modelos.

Capturador:[código](../../../scripts/capturar_monitor_cuda.py). Los logs de sus dos
sesiones están en resultados/paralelo/EXP-003. Se obtuvo autorización para abrir
temporalmente el monitor, que se cerró después de capturar ambas fases.
