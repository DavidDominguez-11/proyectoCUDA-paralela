# Entorno CUDA real y admisión de implementación

Fecha:2026-11-17UTC; PCUDA existente; CAM-006; BIT-008.
Se aplicó compute-env-setup: inspección, ruta local mínima, dependencias ausentes,
smoke real y registro de límites. No se usaron servicios remotos ni credenciales.

## Antes y después

- Windows x64, Python3.14.3, NumPy2.5.5, TensorFlow2.22.0rc0,
  keras-nightly3.16.0.dev2026111503 ya existentes.
- GPU: NVIDIA GeForce GTX1650,4096MiB,sm_75,driver591.59.
- Toolkit encontrado:C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1.
- Se añadieron15paquetes ausentes:7CUDA/Numba y8para ejecutar/validar el notebook.
  Los75paquetes declarados por pip freeze antes de instalar conservaron versión;
  pip25.3 ya existía y se identifica aparte porque freeze lo omite por defecto.
- Evidencia:[verificación de paquetes](verificacion-paquetes.json),
  [antes](paquetes-antes.txt), [después](paquetes-despues.json),
  [versiones fijadas](requirements-pcuda.txt). No se reinstaló NumPy/TensorFlow.

## Comandos de instalación efectivos

Directorio:raíz del repositorio. Los comandos siguientes terminaron0 con uv;
el acceso inicial a su caché fuera del sandbox requirió aprobación de ejecución.

```powershell
uv pip install --python .\PCUDA\Scripts\python.exe numba==0.67.0 llvmlite==0.49.0 numba-cuda==0.30.4 cuda-python==13.5.0 cuda-core==1.2.2 cuda-bindings==13.5.0 cuda-pathfinder==1.9.0
uv pip install --python .\PCUDA\Scripts\python.exe nbclient==0.10.4 nbformat==5.10.4
```

Dependencias añadidas por resolución notebook:attrs26.1.0,fastjsonschema2.22.3,
jsonschema4.26.0,jsonschema-specifications2025.9.1,referencing0.37.0,
rpds-py2026.9.1. No se actualizó pip ni otro paquete anterior.

Antes se ensayó pip install --dry-run, que terminó0 y conservaba NumPy. Dos
intentos reales de descarga con pip quedaron consumiendo CPU/sistema sin generar
salida o instalación completa; se terminaron exclusivamente sus procesos por
PID/comando y se mantuvo el estado previo. Un intento utilizaba legacy-certs.
No se generó el reporte instalacion.json solicitado en esos intentos y no se
presenta como evidencia existente. uv resolvió/descargó/instaló los paquetes.

## Compatibilidad local controlada

La primera compilación falló con AttributeError:numpy.row_stack. numba-cuda0.30.4
todavía registra ese alias, retirado por NumPy2.5. Se define **solo en el proceso**
np.row_stack=np.vstack antes de importar el backend; alias semánticamente idéntico,
sin tocar site-packages ni sustituir NumPy. Se conserva en runtime.py y las pruebas
reales posteriores verifican el comportamiento. No se oculta una caída a CPU.

DLLs de bin/x64 y nvvm/bin/x64 se descubren desde CUDA_HOME13.1. Los handles de
directorios se conservan vivos. El dispositivo real y la compilación sm_75 pasan
la copia de17elementos. CUDA simulator se rechaza explícitamente. Los hashes
cuBLAS y datos device/CPU efectivos quedan en hardware.json de cada RUN.

## Evidencias de corrección

| Evidencia | Estado real y alcance |
|---|---|
| [operadores-002](operadores-002/validacion.json) | Aprobado,51pruebas:conv/Dense fwd/dX/dW/bias,BN,softmax/loss,Dropout,Adam y flagInf |
| [adjuntos-001](adjuntos-001/validacion.json) | Aprobado,78comparaciones locales en formas reales b20/b128, con activaciones/dY iguales |
| [integracion-002](integracion-002/validacion.json) | Aprobado,10pasos sintéticos,último20,contador1172muestras,CPUeval sin mutación,guardado/recarga de pesos+Adam y restore |
| [rutas-004](rutas-004/validacion.json) | Diagnóstico global estricto fallido conservado;1/14cambios de ruta pooling para b2/20,todos dentro de cota de redondeo |

No se declaró aprobada la igualdad estricta de la red completa que falló. La
validación independiente de operadores y sus adjuntos identifica la corrección
de la implementación; la calidad del entrenamiento real se informa por RUN.
Los10pasos sintéticos no miden accuracy ASL: sus etiquetas/datos son artificiales.

Integracion-001 verificó entrenamiento/guardado, pero falló al consultar recursos
compilados:la API numba-cuda0.30.4 recibe objetos firma y no las tuplas de
.signatures. Se corrigió la introspección sin cambiar aritmética; integracion-002
conserva el resultado completo. Los directorios001/002 son distintos, sin sobrescribir.

La variante tiled explícita se mantiene para revisión. El entrenamiento medido
usa im2col/col2im Numba y cuBLAS SGEMM; DEC-022. Las jerarquías internas cuBLAS no
se atribuyen al equipo. El consumo y registros compilados se conservan por RUN.

## Límites

Un entorno local admitido no certifica reconstrucción limpia en otra máquina.
Driver/toolkit/NumPy/Numba y ruedas deben corresponder a las versiones registradas;
un listado de paquetes no garantiza disponibilidad perpetua de una nightly.
No se reinstalaron versiones para forzar resultados. Los binarios grandes y los
datos seleccionados permanecen fuera de Git con rutas/hashes/manifiestos.
