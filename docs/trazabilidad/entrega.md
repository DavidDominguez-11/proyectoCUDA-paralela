# Preparación de la entrega

El PDF oficial exige código documentado e informe PDF. Los Markdown son la
fuente de documentación del proyecto; no sustituyen ese informe final.

**Entrega actual:**[DOCX final](../../entregables/Informe_Proyecto_3_CUDA.docx),
13páginas revisadas,con12tablas,7figuras y los seis apartados en orden. Por
instrucción explícita del usuario,no se exporta aPDF:él realizará la conversión.
[Cobertura y validación](informe-final.md). La saturaciónGPU literal y las
contribuciones verificables del equipo no se certifican mediante el informe.

## Comprobaciones de cierre

- [ ] REQ-01 a REQ-12 tienen evidencia y enlaces válidos.
- [ ] Los tres integrantes tienen contribuciones identificables.
- [x] Código original preservado y versión CUDA documentada.
- [x] Preparación en CPU y entrenamiento con CUDA claramente separados.
- [x] Diseño de hilos/bloques/grids y gestión de memoria sustentado.
- [x] DAT identifica todos los archivos, versiones, splits y transformaciones.
- [ ] Entorno reconstruible y comandos completos por plataforma usada.
- [x] Trabajo equivalente CPU/GPU verificado;diferencias de condiciones documentadas.
- [x] Repeticiones, warmup, restauración de estado y exclusiones documentados.
- [x] Tiempos crudos, promedios y transferencias disponibles.
- [x] Capturas/logs demuestran uso de recursos durante las fases;no saturación literal.
- [x] Corrección numérica/calidad evaluadas con tolerancias y límites declarados.
- [x] Tablas y gráficas pueden regenerarse desde los datos conservados.
- [x] Reproducción desde cero no ejecutada;limitaciones explícitas sin certificarla.
- [x] Informe final DOCX revisado en Word,con autores confirmados y evidencias incrustadas.
- [ ] Informe final exportado a PDF, revisado y vinculado con código/resultados.

## Contenido del informe final

1. Problema, alcance y algoritmo original.
2. Datos utilizados, selección, preparación y particiones.
3. Entorno, hardware y procedimiento experimental.
4. Operaciones repetitivas, oportunidades y diseño paralelo.
5. Implementaciones CPU/CUDA, APIs y cambios realizados.
6. Tiempos por fase, transferencias, variabilidad, speedup y gráficas.
7. Evidencias CPU/GPU con RUN y fase identificables.
8. Respuestas explícitas: por qué preparar archivos/imágenes en CPU y delegar
   matemática a GPU; cómo cambió el tiempo con CUDA; factores que justifican o
   limitan CUDA. Fundamentar en las mediciones y evitar afirmaciones universales.
9. Limitaciones, corrección, conclusiones y contribuciones.
10. Índice de evidencia y pasos de reproducción.

Ruta propuesta: `entregables/informe.md` como fuente y
`entregables/informe.pdf` como exportación final. Crear al existir resultados.
Conservar herramienta/versión/comando de exportación y hash del PDF.
