# RUN-20261117T175407Z-cpu-02

Estado: completada; CPU; EXP-002.

Responsable: Codex por solicitud del usuario; revisión humana pendiente. REQ-03/07/12; PROC-03/04/05/06; DEC-011/012/013; CAM-003; BIT-005.

Pipeline: 2002.551646100 s. Inicio medido UTC: 2026-11-17T17:55:08.368543+00:00; fin: 2026-11-17T18:28:30.920214+00:00.

Carga: 17,400 imágenes; splits12,180/2,610/2,610; cinco redes con15 épocas y batch128, PCA, dos bosques y evaluación/inferencia en15 fotos. Sin descarga ni reinstalación.

El total excluye imports, construcción/warmup, hashes, checkpoints y figuras; fit incluye validación anidada. Caché del SO sin vaciar. CPU6hilos matemáticos, TF inter-op1; RF n_jobs=-1; preparación un worker.

## Evidencias

- [Comando e inicio](invocacion.json).
- [Fin y salida](salida.json).
- [Configuración](configuracion.json).
- [Contexto](contexto/contexto.json).
- [Fuentes exactas](contexto/fuentes.zip).
- [Hardware](hardware.json).
- [Estados iniciales](estados-iniciales.json).
- [Calentamiento](warmup.json).
- [Identidad de tensores](tensores.json).
- [Tiempos crudos](tiempos.csv).
- [Épocas](epocas.csv).
- [Métricas](metricas.json).
- [Recursos](recursos.csv).
- [Salida estándar](stdout.log).
- [Salida de diagnóstico](stderr.log).
- [Modelos locales y hashes](modelos-guardados.json).

La estadística de la serie se publica después de completar todas las repeticiones. Esta observación no es por sí sola una estimación del promedio. No hay resultados GPU en este registro.
