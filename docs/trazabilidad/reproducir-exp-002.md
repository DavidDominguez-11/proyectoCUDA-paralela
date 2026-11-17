# Reproducir las mediciones CPU de EXP-002

Esta guía reutiliza PCUDA y los datos locales. No instala dependencias ni
descarga archivos. Describe una nueva serie; no se ejecutó una sexta corrida
para redactarla. La reconstrucción en otro equipo sigue sin certificarse.

## Evidencia que debe conservarse

- [Configuración ejecutada](../../configuraciones/EXP-002.json).
- [Manifiesto de fuentes](../../resultados/secuencial/EXP-002/manifest_fuente.csv),
  [particiones ordenadas](../../resultados/secuencial/EXP-002/manifest_splits.csv)
  y [mapeo](../../resultados/secuencial/EXP-002/label_mapping.json).
- Cada RUN contiene fuentes exactas en contexto/fuentes.zip, paquetes, hardware,
  estados iniciales, logs, tiempos y manifiesto de integridad.
- Los binarios en data/EXP-002 conservan estados iniciales/modelos de cada RUN.
- [Preservación local de imágenes](../../resultados/secuencial/EXP-002/preservacion-datos.md):
  `data/EXP-002/fuente-seleccionada.zip`, 17,400 imágenes, 224,125,103 bytes;
  SHA-256 `6af5baba0a6049b489263f5b6e159375f1e281ff4c23b61a950d91f557e1c23d`.

data/ está fuera de Git. Conservar esa carpeta con el paquete local de
reproducción o un respaldo. Los manifiestos y hashes solos no recuperan imágenes.

## 1. Preparar una identidad nueva sin volver a muestrear

Desde la raíz del repositorio, con el caché original disponible:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/preparar_reproduccion_cpu.py --origen EXP-002 --nuevo EXP-003
```

El script exige un identificador que no exista; comprueba fuente del benchmark,
notebook, inventario de paquetes y manifiestos, y copia la selección/orden
originales a una configuración nueva. No entrena ni reemplaza EXP-002. Conserva
las identidades DAT originales porque se reutilizan exactamente los mismos datos.

Si cambia el código, restaurar su versión registrada en una copia aislada de
trabajo antes de repetir. El script rechaza una fuente distinta; no restaura
automáticamente archivos ni sobrescribe cambios actuales.

## 2. Recuperar imágenes si ya no está el caché

Verificar primero el SHA-256 del ZIP contra el valor anterior:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath data/EXP-002/fuente-seleccionada.zip
.\PCUDA\Scripts\python.exe -m zipfile -e data/EXP-002/fuente-seleccionada.zip data/EXP-003/entrada
.\PCUDA\Scripts\python.exe -X utf8 scripts/preparar_reproduccion_cpu.py --origen EXP-002 --nuevo EXP-003 --dataset data/EXP-003/entrada
```

Usar una carpeta de extracción nueva. El ZIP contiene solo la selección,
no las 87,000 imágenes originales; no ejecutar `benchmark_cpu.py prepare`
sobre ese subconjunto para volver a sortear datos. El trabajador verifica cada
imagen contra su manifiesto antes de cronometrar.

Las opciones 1 y 2 son alternativas: preparar EXP-003 una sola vez.

## 3. Ejecutar cinco corridas y el supervisor de cierre

En una terminal PowerShell situada en la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/benchmark_cpu.py run --experimento EXP-003
```

En otra terminal, también desde la raíz:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/cerrar_serie_cpu.py --experimento EXP-003
```

El supervisor aplica DEC-015: espera cinco trabajadores con salida 0 y detiene
el controlador antes de ampliar la serie según un total adaptativo mezclado.
El controlador puede finalizar por terminación deliberada; revisar las cinco
salidas, serie.json y cierre-alcance.md en lugar de interpretar ese código
del controlador como fallo de un entrenamiento. Se conserva su estado previo.

No ejecutar otras tareas de cómputo deliberadamente durante la serie. Registrar
el plan de energía y cualquier diferencia de hardware/entorno; los tiempos no
son trasladables automáticamente a otra máquina.

## 4. Cerrar evidencias y regenerar el análisis

Después de terminar la serie:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/sellar_corridas_cpu.py --experimento EXP-003
.\PCUDA\Scripts\python.exe -X utf8 scripts/preservar_datos_cpu.py --experimento EXP-003
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cpu.py --experimento EXP-003
```

Para regenerar solamente tablas y gráficas de la serie existente:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/resumir_cpu.py --experimento EXP-002
```

El análisis valida hashes de checkpoints iniciales/finales, contadores Adam,
manifiestos sellados, paquetes, fotos y tensores. La referencia suma las 14 fases
comunes; las ramas variables se estratifican. n=1 implica DE/IC no aplicables.

## Límites comprobados

Se reprodujeron datos, estados iniciales y procedimiento entre las cinco
ejecuciones. No se reprodujo la trayectoria de aprendizaje de CNN-B dentro de
1e-6 y cambió una selección de arquitectura. No hay garantía de resultados
numéricos o tiempos idénticos al repetir. Es un límite observado, no un fallo
que se haya ocultado cambiando dependencias o eliminando corridas.
