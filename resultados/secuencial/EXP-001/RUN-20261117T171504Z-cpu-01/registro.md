# RUN-20261117T171504Z-cpu-01 — interrumpida

Estado: fallida por instrumentacion; excluida completa del promedio. EXP-001; BIT-005; CAM-003.

El monitor seguia el lanzador Windows del venv, no su hijo TensorFlow. CPU del proceso y RSS no describen al trabajador; los datos globales por nucleo son reales pero la corrida quedo incompleta. Se termino el proceso propio para corregir y comenzar EXP-002.

Los flags de fases completadas de tiempos.csv fueron escritos antes de interrumpir; no autorizan usar esta corrida como repeticion completa. No se alteran retrospectivamente esos datos.

- [Incidencia](incidencia-monitor.md).
- [Invocacion](invocacion.json), [salida](salida.json).
- [Configuracion](configuracion.json), [contexto](contexto/contexto.json).
- [stdout](stdout.log), [stderr](stderr.log), [tiempos parciales](tiempos.csv), [monitor original](recursos.csv).
