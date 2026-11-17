# Preservación local de datos de EXP-002

Fecha UTC: 2026-11-17T21:34:46.779443+00:00. Archivo: `data/EXP-002/fuente-seleccionada.zip`.

Imágenes: 17400; bytes: 224125103; SHA-256: `6af5baba0a6049b489263f5b6e159375f1e281ff4c23b61a950d91f557e1c23d`.

Se verificó cada entrada contra manifest_fuente.csv y la integridad del ZIP. Copia local sin descarga ni modificación del caché, creada después de medir. Entradas ordenadas y metadatos ZIP fijos para evitar timestamps variables en el archivo.

El ZIP conserva únicamente las imágenes seleccionadas, con rutas relativas originales; no sustituye al inventario de 87,000 archivos del proveedor. Las 15 fotos propias están en las fuentes de cada contexto.

Comando desde la raíz (rechaza sobrescribir el ZIP):

```powershell
.\PCUDA\Scripts\python.exe -X utf8 scripts/preservar_datos_cpu.py --experimento EXP-002
```

Para recuperar entradas sin red: verificar SHA-256, extraer el ZIP a una carpeta nueva y reutilizar manifest_splits.csv/label_mapping.json; no volver a muestrear. Registrar una configuración nueva si cambia la ruta del dataset. No ejecutar prepare sobre el subconjunto como si contuviera las 87,000 imágenes.

Los binarios de data/ están ignorados por Git. Conservarlos con el paquete local de reproducción o una copia de respaldo; publicar sus manifiestos/hashes. No se certifica recuperación en otra máquina ni reconstrucción de PCUDA desde cero.
