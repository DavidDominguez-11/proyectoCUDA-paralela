# Configuraciones de experimentos

| Configuración | Estado / alcance |
|---|---|
| [EXP-001](EXP-001.json) | Intento CPU interrumpido por monitorización del lanzador; preservado, sin promedio válido |
| [EXP-002](EXP-002.json) | Cinco corridas CPU completadas; monitor corregido; referencia común y ramas separadas mediante DEC-015 |

`experimento.plantilla.json` describe los campos que deberán completarse al
definir un EXP; no es consumida por el notebook actual ni habilita CUDA.

1. Copiar a `EXP-NNN.json` sin sobrescribir la plantilla.
2. Registrar decisiones, datos/hashes, entorno, carga común y comandos CPU/GPU.
3. Resolver campos desconocidos antes de medir. Para un campo opcional no
   aplicable, registrar explícitamente `"no_aplica"` y el motivo en la decisión;
   `null` identifica información pendiente. Mantener el tipo adecuado de cada
   parámetro numérico o booleano que realmente use la implementación.
4. Cambiar `es_plantilla` a false al completar y revisar. Conservar el SHA-256 del
   archivo exacto en cada RUN; copiarlo allí antes de ejecutar.
5. Implementar lectura/validación de parámetros en el código experimental.
   Una configuración adjunta que el programa ignora no prueba equivalencia.
6. Cualquier modificación posterior genera otra versión de configuración y su
   registro CAM/DEC; no reutilizar silenciosamente un identificador congelado.

Las diferencias específicas CPU/GPU coexisten en el mismo documento común;
documentar cuáles son inevitables y cuáles afectan comparabilidad. El esquema
se puede ampliar con decisiones justificadas, conservando las versiones usadas.
