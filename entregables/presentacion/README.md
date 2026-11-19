# Presentación del Proyecto3CUDA

[Abrir PowerPoint editable](../Presentacion_Proyecto_3_CUDA.pptx) · [Guion para exponer](Guion_de_exposicion.md) · [Vista general](Vista_general.png)

31diapositivas en16:9:26principales y5anexos. Diagramas del flujo original yCPU/GPU,operaciones repetitivas,referenciaCPU,selección,diseñoCUDA,implementación,pruebas,capturas reales,tiempos,speedup,variabilidad,curvas,calidad,limitaciones yconclusiones. Tablas y13gráficas nativas editables,31notas del presentador. Las capturas se copian sin alterar y las muestrasASL pertenecen a los archivos reales del manifiesto. No contiene imágenes generadas ni benchmarks nuevos.

## Uso durante la exposición

- Exponer las diapositivas1–26 y conservar27–31 para preguntas.
- Usar Vista del moderador para leer las notas; el guionMarkdown contiene la misma explicación.
- Resultados principales:17–23. Evidencia de dispositivos:14–16. Conclusión:26.
- En17 las escalas son diferentes por arquitectura; comparar CPU/CUDA dentro del panel.
- En18 cada columna es la secuencia de cinco repeticiones de su campaña; una misma fila no significa par CPU/GPU simultáneo.
- En20 se muestran proporciones normalizadas y una tabla en segundos; no sumar los componentes otra vez al total.
- En23 separar la rapidez de la calidad: lasFC bajaron1.34/4.33puntos porcentuales frente aCPUhistórica.

## Fuentes y reproducción

Datos usados en datos-presentacion.json;39archivos originales conSHA-256 en fuentes.csv. Guion/notas en estructura-y-notas.json y Guion_de_exposicion.md. Generador:scripts/crear_presentacion_cuda.cjs,preparación:scripts/preparar_presentacion.py. Las dependenciasNode se añadieron únicamente a data/presentacion-tooling;PCUDA permanece intacto. Se adjuntan manifiestospackage.json/package-lock para fijar las versiones usadas.

El PPTX incorpora las imágenes y hojas de datos; puede abrirse sin ejecutar ningún notebook ni disponer deldataset. Para reconstruir desde este repositorio ya preparado:

```powershell
node scripts/crear_presentacion_cuda.cjs
python -X utf8 scripts/revisar_presentacion_cuda.py --revision revision-NUEVA
```

El generador recompone elPPTX:conservar una copia antes si se hacen ediciones manuales. El preparador inicial rechaza destinos existentes. La revisión no exportaPDF:abre conPowerPointlocalenlectura,sacaPNG yverificaarchivo,relaciones,ejes,colores,límites ynotas. Los auxiliares sugeridos porSKILL-P.md no estaban incluidos;se realizaron controles equivalentes locales. El generador corrige una referencia de eje serie no declarada quepptxgenjs4.0.1 emite en gráficos2D. No se modifican los datos al hacerlo.

## Reservas importantes

CPUhistórica,nocinco pares alternados;shuffle/Dropout distintos,misma arquitectura/datos/estado inicial/pasos. Preparación1→6workers esCPU;testcontinúaCPU. ActividadGPUnoequivaleaocupaciónSMnisaturaciónVRAM. “Secuencial”es la referencia del proyecto,conparalelismo internoCPUhabilitado. El contenido diferenciamediciones,cálculosderivados ehipótesis. Se conservan los documentos ynotebooks previos sinningúncambio.


[Manifiesto SHA-256 de entrega](artefactos-v02.csv). Las copias previas de trazabilidad se conservan con extensión.md.bak para distinguirlas de documentación activa.
