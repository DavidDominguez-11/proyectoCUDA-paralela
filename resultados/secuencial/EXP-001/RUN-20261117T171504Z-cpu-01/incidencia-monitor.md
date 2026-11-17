# Corrida interrumpida por monitorizacion incorrecta

UTC: 2026-11-17T17:17:29.509067+00:00

El monitor capturaba el lanzador del venv (RSS ~4 MiB, CPU0), no el proceso hijo TensorFlow. Se conserva la evidencia parcial sin usarla como benchmark completo. Se detiene esta serie y se prepara otra con monitor de arbol de procesos.
