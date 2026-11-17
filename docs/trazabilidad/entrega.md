# Preparación de la entrega

El PDF oficial exige código documentado e informe PDF. Los Markdown son la
fuente de documentación del proyecto; no sustituyen ese informe final.

## Comprobaciones de cierre

- [ ] REQ-01 a REQ-12 tienen evidencia y enlaces válidos.
- [ ] Los tres integrantes tienen contribuciones identificables.
- [ ] Código original preservado y versión CUDA documentada.
- [ ] Preparación en CPU y entrenamiento con CUDA claramente separados.
- [ ] Diseño de hilos/bloques/grids y gestión de memoria sustentado.
- [ ] DAT identifica todos los archivos, versiones, splits y transformaciones.
- [ ] Entorno reconstruible y comandos completos por plataforma usada.
- [ ] Configuración común CPU/GPU y trabajo equivalente verificados.
- [ ] Repeticiones, warmup, restauración de estado y exclusiones documentados.
- [ ] Tiempos crudos, promedios y transferencias disponibles.
- [ ] Capturas/logs demuestran uso de recursos durante las fases.
- [ ] Corrección numérica/calidad evaluadas con tolerancias declaradas.
- [ ] Tablas y gráficas pueden regenerarse desde los datos conservados.
- [ ] Reproducción desde cero registrada, o limitaciones explícitas sin certificarla.
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
