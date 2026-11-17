# Evidencia de inspección para el diseño del paso 4

Tipo: **inventario observado, inspección estática y cálculos derivados**.
Fecha: 2026-11-17 UTC. Responsable: Codex. DIS-CUDA-001; BIT-007; CAM-005.
No es un benchmark, prueba de kernels ni evidencia de utilización GPU bajo carga.
No se crea EXP/RUN y no se modifican archivos sellados de EXP-002.

## 1. Estado previo e identidades

Árbol limpio antes de las modificaciones; commit consultado con
`git log -1 --format=%H`:
`2a14000a54fb38bc72dc9a45d5438ca726ae76bf`.
Este commit identifica la versión previa de los documentos existentes de
trazabilidad que actualiza CAM-005. Los documentos nuevos del diseño
no tienen identidad previa. El original permanece preservado.

| Archivo de entrada | SHA-256 observado |
|---|---|
| docs/Proyecto 3 CUDA.pdf | `2dd8efc1ffb36d1b0908fbb400808a713b99e1c1322f71843e3e22198f72eb6d` (identidad del inventario, revalidada por el verificador) |
| LAB3-DATASCIENCE/Lab3_ASL_Deep_Learning.ipynb | `ce8824e860ae7e02684bf702fd6865b7d87cc9eb71917f56bfb0ee8d9ccb8e8a` (identidad del inventario, revalidada por el verificador) |
| docs/trazabilidad/estrategia-paralelizacion.md antes de CAM-005 | `8e1a250ade67268cce47898e7b8442d0d3c9b25d74c048242e360c358ae7ccc3` |
| docs/trazabilidad/oportunidades-paralelizacion.md | `6cffbf6fed14d5562a392ffb30e5bd6e1aeffdb8cb0852a5ded270867af89941` |
| PCUDA/Lib/site-packages/keras/src/optimizers/adam.py | `959715bc19e4670a5d7a3d65b72889e55644e4ca867810099b6f9f85d85ab91b` |
| PCUDA/Lib/site-packages/keras/src/layers/normalization/batch_normalization.py | `b81aa81c666f2e35c3d9734e1fea074c1922159d06cc5aceee60db7895f57a35` |
| data/EXP-002/RUN-20261117T171942Z-cpu-01/CNN-B.estado-inicial.npz | `10558d723213752b759626c3b196665f2547a0b65cbce296e47b3a4f992305fd` |

Los archivos PCUDA y el checkpoint grande son locales fuera de Git; no se
presentan como vínculos portables. Su función, forma y contrato se documentan en
el diseño; el checkpoint ya está referenciado en los manifiestos de EXP-002.

## 2. Consulta de dispositivo: inventario real

Desde la raíz del repositorio, PowerShell, consulta repetida con fecha explícita:

```powershell
[DateTime]::UtcNow.ToString('o')
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap --format=csv,noheader
```

Salida, código0:

```text
2026-11-17T22:17:26.5565353Z
NVIDIA GeForce GTX 1650, GPU-bc6b00a7-4246-f800-69e3-0b3e2c7ec995, 591.59, 4096 MiB, 3935 MiB, 7.5
```

La memoria libre corresponde solo a ese instante. Este comando no identifica
toolkit/NVVM instalado, compilación Numba, ocupación de SM ni entrenamiento real.
La consulta previa de la misma sesión devolvió los mismos valores; no se usa
como otra repetición experimental ni como certificación de disponibilidad futura.

## 3. Metadatos del entorno, sin instalación

Lectura con `importlib.metadata` en PCUDA; no se importó TensorFlow para construir
o ejecutar modelos. Consulta realizada con el ejecutable existente:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 -c "import importlib.metadata as m,sys; print('Python',sys.version.split()[0]); print([(x,next((d.version for d in m.distributions() if d.metadata['Name'].lower().replace('_','-')==x),'NO INSTALADO')) for x in ['numpy','tensorflow','keras-nightly','numba','numba-cuda','cuda-python']])"
```

El bloque anterior reproduce la consulta de metadatos; en la sesión se ejecutó
también junto con cálculos de payload y hash del checkpoint. Salida observada:

```text
Python 3.14.3
[('numpy', '2.5.5'), ('tensorflow', '2.22.0rc0'), ('keras-nightly', '3.16.0.dev2026111503'), ('numba', 'NO INSTALADO'), ('numba-cuda', 'NO INSTALADO'), ('cuda-python', 'NO INSTALADO')]
```

No se ha certificado la compatibilidad de un stack CUDA nuevo con PCUDA. No se
instalaron ni reinstalaron dependencias, ni se descargaron datos o modelos.

## Presentación complementaria conservada

La presentación de Semana15 añadida por el usuario se leyó de su ruta externa
y se preservó en [docs/referencias](../../docs/referencias/semana-15-programacion-cuda.md)
por copia binaria con creación exclusiva, sin sobrescribir.16005bytes,341líneas,
SHA-256 acce03142813c9aca0bf4329000639bb2bb986f92e2810bd1c291774e73fab26.
El [contraste](../../docs/trazabilidad/contraste-semana-15.md) registra su función
complementaria y el ajuste de lectura por canales, sin atribuirle precedencia.

## 4. Inspección estática y lectura normativa

Se leyeron como JSON las celdas24/31/32/33/34 del notebook, sin ejecutarlas.
Se contrastaron Adam.update_step y BatchNormalization.call/_moments en fuentes
locales. Observaciones relevantes: Conv tiene ReLU antes de BN; cinco Conv,
cuatro BN/cuatro Dropout; epsilon BN0.001; momentum0.9; Adam epsilon1e-7 fuera
de sqrt(v), con alpha corregida por bias temporal.

Se leyó también el checkpoint NPZ con NumPy, allow_pickle=False, sin construir
modelo:80 arreglos,75 flotantes con6736696 valores; cuatro estados RNG int64[2]
en arr_8/17/24/31 y contador Adam escalar int64 en arr_34, valor0. El índice de
variables se contrastó con la construcción de la instantánea en benchmark_cpu.py
(model.variables seguido de optimizer.variables), sin modificar ese script.
Esta lectura detectó que es necesario preservar los estados RNG como procedencia
y declarar su sustitución por máscaras compartidas; no contabilizarlos como pesos.

La página3 del PDF se extrajo con Python del sistema, pypdf, código0:

```powershell
python -X utf8 -c "from pypdf import PdfReader; r=PdfReader('docs/Proyecto 3 CUDA.pdf'); print(r.pages[2].extract_text())"
```

Para leer BN se intentó localizar `elif training` mediante str.index; no existe
esa cadena y se obtuvo **ValueError: substring not found**, código1. No hubo
mutación ni ejecución de modelo. Se corrigió la consulta a `if training and
self.trainable`, obteniendo el bloque y las actualizaciones móviles, código0:

```powershell
.\PCUDA\Scripts\python.exe -X utf8 -c "from pathlib import Path; p=Path('PCUDA/Lib/site-packages/keras/src/layers/normalization/batch_normalization.py'); s=p.read_text(encoding='utf8'); a=s.index('if training and self.trainable'); print(s[a:a+3300]);"
```

## 5. Cálculos derivados, no memoria ni tráfico medidos

| Concepto | Fórmula / resultado |
|---|---|
| Parámetros Conv | sum(9CiCo+Co) =139424 |
| Parámetros Dense | 8192×256+256+256×29+29 =2104861 |
| Gamma/beta BN | 2×(32+64+128+256) =960 |
| Parámetros entrenables | 2245245; 8.564930 MiB FP32 |
| Estado BN móvil | 960 valores; modelo total2246205 valores |
| Parámetros + gradientes + m/v + BN | (4×2245245+960)×4 =34.263382 MiB |
| Activaciones separadas | Suma de todas las formas de la tabla arquitectónica, entrada/logits/probs incluidos; Flatten vista =382.403320 MiB en b128 |
| Máscaras por muestra | 32768+16384+8192+256 =57600 bytes uint8 |
| Máscaras por lote128 | 7.03125 MiB |
| Índices pooling | 128×(32768+16384+8192) =7 MiB uint8 |
| Mayor parcial dW | C2 o C4:9 MiB; arena reservada16 MiB |
| Buffers presupuestados | 382.403320+128+34.263382+16+4+7.03125+7+1 =679.697952 MiB; tope768 MiB |
| Imágenes H2D/época | 12180×12288×4 =598671360 bytes |
| Máscaras H2D/época | 12180×57600 =701568000 bytes |
| Imágenes+masks+labels H2D/15épocas | 15×12180×(12288×4+57600+8) =19505052000 bytes; no incluye estado inicial |
| Modelo D2H/época | 2246205×4 =8984820 bytes |

Los cálculos se realizaron con enteros de Python/operaciones de tamaño, sin
reservar tensores GPU ni entrenar. Las fórmulas permiten recalcular la tabla sin
paquetes CUDA. No incluyen contexto, cachés de código ni tráfico físico del bus.

## 6. Fuentes técnicas consultadas

Consulta web el2026-11-17, exclusivamente documentación primaria, como apoyo de
API/modelo de ejecución; no sustituye al PDF del proyecto:

- [NVIDIA CUDA: modelo de programación](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).
- [NVIDIA Numba-CUDA: instalación](https://nvidia.github.io/numba-cuda/user/installation.html),
  [kernels Python](https://nvidia.github.io/numba-cuda/user/kernels.html),
  [API kernel](https://nvidia.github.io/numba-cuda/reference/kernel.html),
  [memoria](https://nvidia.github.io/numba-cuda/user/memory.html) y
  [eventos/API Host](https://nvidia.github.io/numba-cuda/reference/host.html).
- [Keras BN](https://keras.io/api/layers/normalization_layers/batch_normalization/) y
  [Adam](https://keras.io/api/optimizers/adam/), contrastados con archivos instalados.
- [NumPy PCG64](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html) y
  [SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html).

Las versiones de paquetes efectivamente instaladas, toolchain y recursos
compilados deberán quedar en el lock/contexto de los RUN futuros. Una página web
actual y una propuesta algorítmica no prueban compatibilidad binaria del entorno.

La validación de enlaces, originales e integridad documental se registra en
BIT-007 y el [reporte mutable de preparación](../preparacion/verificacion-preparacion.md).
