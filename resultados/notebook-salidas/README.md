# Notebook con salidas reales incrustadas

Entregable:[Lab3_ASL_CUDA_con_salidas.ipynb](../../notebooks/Lab3_ASL_CUDA_con_salidas.ipynb).
Solicitud posterior del usuario:dejar los resultados reales dentro del notebook,
al menos los de la última corrida. CAM-007; BIT-009. No es otro experimento.

Origen:RUN-20261117T234658Z-gpu-05 de EXP-003,terminado con código0.
Se muestran al principio del notebook las15líneas EPOCH originales de stdout.log,
tabla completa epocas.csv,métricas test/validación,tiempos de pared/eventos y
transferencias,curvas de esa corrida,hardware y hashes de procedencia.

Se ejecutaron44celdas sin errores y quedaron5imágenes incrustadas (curva de la
última corrida más cuatro gráficas agregadas). El código original protegido y
la implementación documentada siguen incluidos. El notebook es de consulta:
no vuelve a entrenar ni inicializa la GPU para mostrar hardware archivado.

Prueba:[salida del proceso](salida.json) y [validación](validacion.json).
La serie mantiene exactamente su SHA-256 antes/después;las184evidencias del
manifiesto sellado EXP-003 se verificaron sin cambios. El notebook fuente
congelado conserva su identidad53050fbcc27a8629850d36165c5c4387982714277f09a58e256e3537b051b86f.

Para reproducir la visualización se usa [plantilla.ipynb](plantilla.ipynb),
con otro destino de salida que no exista:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/ejecutar_notebook_cuda.py --entrada resultados/notebook-salidas/plantilla.ipynb --salida notebooks/otra-vista-con-salidas.ipynb
```

Ese ejemplo es de consulta,no una medición nueva. La plantilla fuerza
EJECUTAR_GPU=False y lee los archivos originales. No se presentan los logs
recuperados como si el entrenamiento acabara de repetirse. Los comandos reales
de esta entrega están en salida.json;stdout/stderr se conservan junto a ella.
