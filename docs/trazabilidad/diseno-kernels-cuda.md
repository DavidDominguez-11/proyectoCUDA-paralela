# DIS-CUDA-001. Catálogo de kernels y jerarquía de ejecución

**Antecedente revisión1.** En [paso5](implementacion-cuda.md), DEC-022 conserva
la variante tiled como referencia y usa cuBLAS para productos grandes con
im2col/col2im explícitos. Las jerarquías propias y las del backend se distinguen
en código,lanzamientos.json y reporte; no se atribuye a cuBLAS el grid16×16 propuesto.

Parte del [diseño del paso 4](diseno-paralelo.md), revisión1. **Especificación,
sin implementar ni ejecutar**. Los nombres K01–K19 identifican contratos futuros,
no funciones existentes. Se cubren OP-P01..OP-P09 del paso3.

## A. Convenciones comunes y límites

- `ceildiv(a,b) = (a+b-1)//b` para enteros positivos. No lanzar grids vacíos.
- NHWC lineal: `i = (((n*H+h)*W+w)*C+c)`. Decodificar con división/resto enteros,
  primero c, luego w,h,n. No confundir canal con índice de imagen.
- Todos los offsets y productos de dimensiones se evalúan en int64. Una máscara
  uint8 vale exactamente0 o1. El contador global Adam es int64 y comienza en0.
- G1D(E): block=(256,1,1), grid=(min(4096,ceildiv(E,256)),1,1). Thread t empieza
  en blockIdx.x×256+threadIdx.x y recorre i+=256×gridDim.x mientras i<E.
  Cada i tiene un único escritor; para E no múltiplo de256 no se accede a la cola.
- G16(M,N): block=(16,16,1). Thread(tx,ty) produce fila
  `m=m0+16*blockIdx.y+ty`, columna `n=16*blockIdx.x+tx`.
  Dividir M en slabs consecutivos de máximo**16384 filas**, m0 múltiplo de16;
  grid físico=(ceildiv(N,16),ceildiv(min(16384,M-m0),16),1).
  Grid lógico completo=(ceildiv(N,16),ceildiv(M,16),1). Procesar todos los slabs,
  sin cambiar el resultado ni mezclar lotes. Es una cota de trabajo por lanzamiento,
  no evidencia de tiempo o garantía contra watchdog.
- G16S(M,N,K): igual a G16, con grid.z=ceildiv(K,2048). Cada z reduce únicamente
  k desde2048z hasta min(K,2048(z+1)); escribe un parcial propio [z,m,n].
  Los slabs de M son los mismos. No atomics flotantes; K04 reúne los parciales.
- Reducción R(L,C): block=(32,8,1), grid=(ceildiv(L,2048),ceildiv(C,32),1).
  Thread(tx,ty) atiende canal c=32*blockIdx.y+tx y filas
  l=2048p+ty+8j, j=0..255, p=blockIdx.x; fuera de L/C aporta0.
  Cada warp recorre32canales contiguos de una misma fila NHWC. Suma local en
  orden j; escribe shared[ty,tx]. Árbol sobre ty con strides4,2,1 y barrera
  uniforme en cada nivel. Solo ty=0 y c<C escribe parcial[p,c]. Hay hasta256 valores
  por thread; se acepta el coste serial para mantener lectura contigua sin
  materializar transpuestas. Es una propuesta sin rendimiento medido.
- Fusión RF(P,C): block256, grid=(C,1,1). Thread t lee parcial p=t si t<P;
  como P≤256 en este alcance basta un valor por lane. Árbol256 con ceros en la
  cola; t=0 escribe salida. Si se excediera P=256, rechazar forma: no truncar.
- Las barreras son de block y todas sus lanes participan. Un thread fuera de
  forma **no retorna antes de una barrera**: carga neutros y omite escritura final.
  Las fronteras de kernels en el stream único son las barreras entre blocks.

La jerarquía propuesta utiliza exclusivamente blocks de1,32 o256 threads.
Los índices de Numba se rigen por su [API oficial de kernels](https://nvidia.github.io/numba-cuda/reference/kernel.html).
Consultar límites efectivos del dispositivo antes de lanzar; la planificación
de warps, registros físicos y SM es responsabilidad del compilador/driver.

En un block16×16 el índice lineal del thread es tx+16ty: cada warp ocupa dos
filas de16 threads. En32×8 el índice es tx+32ty: cada warp ocupa una fila de32
canales. Ninguno de los algoritmos depende de sincronización implícita entre
lanes: mantiene barreras de block explícitas incluso en reducciones de un warp.

### Multiplicación matricial común sin materializaciones gigantes

Para `C[M,N] = A[M,K] · B[K,N]`, cada block G16 carga cooperativamente dos tiles
shared float32[16,16]: A[fila,k0+tx] y B[k0+ty,col]. El acceso A/B puede resolver
un elemento de una matriz implícita mediante las fórmulas de abajo. Fuera de
límites se carga0. Tras barrera, cada thread acumula16 productos en orden k=0..15;
barrera antes de reemplazar los tiles; avanzar k0 en16 hasta K o el fin de su
partición. Un acumulador float32 por salida, suma/multiplicación FP32 explícitas.

Shared por block: **2048 bytes**. Tiles y tamaños son constantes de compilación,
no arreglos dinámicos locales por thread. No se materializa im2col ni matrices
transpuestas. Se consulta la información de registros/shared/local del compilador;
no se afirma que un valor local se aloje siempre en registro ni se limita registros
de manera arbitraria. La indexación implícita puede producir lecturas dispersas,
especialmente en dX; no se promete coalescencia perfecta.

## B. Convolución: K01, K02, K03 y K04

Notación: X[b,H,W,Ci], Wc[3,3,Ci,Co], bias[Co], Z[b,H,W,Co]. Stride1 y pad1.
El original realiza correlación cruzada, sin invertir filtros en forward.

### K01 — Forward convolucional + bias + ReLU (OP-P01/04)

- Matrices implícitas: M=bHW, N=Co, K=9Ci; usar G16(M,N).
- Fila m se decodifica en(n,h,w). k se decodifica en(kh,kw,ci), ci contiguo.
  A[m,k]=X[n,h+kh-1,w+kw-1,ci] o0 si sale de imagen;
  B[k,co]=Wc[kh,kw,ci,co].
- Salida U[m,co]=max(0,sum_k A·B+bias[co]). Conservar U para derivada ReLU y
  para la siguiente capa; en las capas C1/C3/C5, BN lee U después de ReLU.
- Dueño único de U[m,co]; no compartición de escritura. Fuera de forma no escribe.
- Bias se suma una sola vez tras reducción. Para u exactamente0 se adopta derivada0.
  No aplicar ReLU en la última Dense de logits (K05).

### K02 — Gradiente de entrada convolucional (OP-P03)

Sea D[n,ho,wo,co] el gradiente **respecto a preactivación**, después de aplicar
la derivada ReLU a su gradiente recibido. Se calcula:

`dX[n,hi,wi,ci] = sum_(kh,kw,co) D[n,hi-kh+1,wi-kw+1,co] * Wc[kh,kw,ci,co]`.

- M=bHW, N=Ci, K=9Co, G16(M,N). Decodificar fila a(n,hi,wi) y k a(kh,kw,co).
- A[m,k] es D con esos offsets o0 fuera de rango; B[k,ci] es el filtro con
  índices anteriores. Esta es la inversión espacial implícita necesaria en dX;
  no repetir erróneamente los offsets forward.
- Un thread produce una entrada completa dX: gather, sin scatter ni atomics.
  No se escribe sobre D ni Wc. **Omitir dX de C1**, porque la imagen no es
  entrenable y no existe capa previa; sí calcular dW/db de C1.

### K03 — Gradiente de pesos convolucional, parciales (OP-P03)

`dWc[kh,kw,ci,co] = sum_(n,h,w) X[n,h+kh-1,w+kw-1,ci] * D[n,h,w,co]`.

- Producto virtual A^T·D: M=9Ci, N=Co, K=bHW. G16S(M,N,K), porción2048 de K.
- Fila m decodifica kh,kw,ci; índice de reducción k decodifica n,h,w.
  A[m,k]=entrada X con padding; B[k,co]=D. Cada thread escribe
  parcial[z,m,co] en arena global reutilizable.
- Padding virtual aporta0; ni lecturas fuera de imagen ni accesos del último
  lote usan el tamaño máximo del buffer como tamaño lógico.
- No actualizar Wc ni acumular sobre un gradiente anterior; cada parcial se
  sobrescribe. Peso definitivo se obtiene con K04.

### K04 — Fusión determinista de parciales de pesos (OP-P03)

G1D(E), E=M×N pesos reales. Cada thread reduce parcial[z,i] con z ascendente,
desde0 hasta P-1, acumulador FP32; escribe grad[i]. No añade otro promedio:
el factor1/b ya aparece en dLogits. No lee padding de tiles. El workspace se
reutiliza solo después de finalizar K04. Para P=1 se conserva la misma operación
o se escribe directamente el gradiente, siempre registrado como especialización.

## C. Capas densas: K05–K07

X[b,I], Wd[I,O], bias[O]. El Flatten NHWC es una vista b×8192 con idéntico
orden de elementos; backward recupera la forma anterior sin copiar ni permutar.

| Kernel / OP | Ecuación | Jerarquía y propiedad de escritura |
|---|---|---|
| K05 forward / OP-P02/04 | Y=X·Wd+bias; F1 aplica ReLU, Z no la aplica | G16(b,O), reducción K=I; una salida por thread, bias una vez |
| K06 dX / OP-P03 | dX=D·Wd^T | G16(b,I), K=O; W transpuesta solo al acceder; conserva pesos anteriores |
| K07 dW / OP-P03 | dWd=X^T·D | G16S(I,O,b); b≤128 implica P=1; K04 o escritura directa según especialización registrada |

K05–K07 comparten tiles2048 bytes; flags de ReLU son especializaciones distintas
para F1 y Z, no dependen de inferir el tipo de capa por la forma. Los bias de
Conv y Dense utilizan K08. Todos los gradientes de parámetros se calculan antes
de ejecutar cualquier K17 de Adam.

## D. Reducciones de bias: K08

Para D[L,C], con L=bHW en Conv y L=b en Dense:
`db[c] = sum_l D[l,c]`. Aplicar R(L,C) con un shared[8,32]float32 (1024 bytes),
seguido de RF(P,C) con shared[256]. El resultado final escribe el gradiente db de ese paso,
incluidos bias de convoluciones seguidas de BN. No omitirlos por considerarlos
redundantes: existen en la arquitectura original. No normalizar otra vez por L.

## E. ReLU y Dropout: K09 y K10

### K09 — Derivada ReLU (OP-P04)

G1D(E), E=tamaño de activación. `dZ[i]=dU[i]` si U[i]>0, y0 en otro caso.
U es la salida ReLU conservada. Se permite transformar el buffer dU in-place
porque cada i solo lee/escribe ese mismo i; U permanece aparte e intacto. La
entrada real de BN no debe sobrescribirse antes de su propio backward.

### K10 — Dropout forward/backward (OP-P09)

G1D(E). Forward Y[i]=X[i]×mask[i]/(1-p). Backward
dX[i]=dY[i]×mask[i]/(1-p). Misma máscara uint8, mismo factor FP32 y orden de
operaciones en ambos dispositivos. El cálculo 1/(1-p) se convierte una vez a FP32
y se congela; se usa multiplicación por ese factor en ambos recorridos.

La CPU genera cada máscara conforme al contrato de aleatoriedad; no se sustituye
el generador por compartir solo el número de semilla. No aplica máscara compartida
por canal, ni escalado por b, ni máscara nueva en backward. En validación CPU
Dropout es identidad y no consume números aleatorios. Forward no es in-place;
backward puede serlo sobre el gradiente por la misma regla que ReLU.

## F. MaxPooling: K11 y K12 (OP-P05)

X[b,H,W,C], H/W pares en CNN-B, Y[b,H/2,W/2,C].

### K11 — Forward

G1D(b×H/2×W/2×C). Un thread decodifica n,ho,wo,c, examina
X[n,2ho+dh,2wo+dw,c], dh/dw=0..1 en orden(0,0),(0,1),(1,0),(1,1).
Actualiza máximo solo si el nuevo valor es estrictamente mayor. Escribe Y y
argmax uint8=2dh+dw. No usa shared ni requiere sincronización. Inicializar con
el primer valor, no con0, para permitir entradas negativas tras BN.

### K12 — Backward gather

G1D(bHWC). Un thread por entrada n,h,w,c obtiene ho=h//2, wo=w//2 y posición
local2(h%2)+(w%2). Escribe dX=dY[n,ho,wo,c] si coincide con argmax, y0 en caso
contrario. Las ventanas2×2 stride2 no se solapan: no hay suma concurrente ni se
necesita atomicAdd. No extender esta regla a pooling solapado sin rediseñar.

El desempate se verifica frente a la referencia CPU con ventanas empatadas;
si el backend usa otra convención, se corrige de manera explícita antes de medir.
No se omite la prueba porque los tensores reales rara vez empaten.

## G. BatchNormalization: K13, K14 y K15 (OP-P06)

Interpretar entrada U como [L,C], L=bHW para B1/B2/B3, L=b para B4.
Entrenamiento usa varianza **poblacional** (divisor L), no corrección L-1.

### K13 — Estadísticas en cuatro lanzamientos

1. R(L,C) sobre U para parciales de suma; RF(P,C) suma parciales y divide por L,
   produciendo mu[c]. Shared1024 bytes por block en ambos lanzamientos;
   forma[8,32] en R y[256] en RF.
2. R(L,C) sobre `(U-mu[c])²`; RF(P,C) divide por L y produce var[c]. Se usan
   dos pasadas para evitar la cancelación de E[x²]-E[x]². Redondeos no garantizan
   coincidencia bit a bit con el backend CPU; sí deben cumplir sus pruebas.
3. El último pase calcula invstd[c]=1/sqrt(var[c]+0.001), y actualiza **una vez**
   moving_mean=.9×moving_mean+.1×mu y moving_var=.9×moving_var+.1×var.
   Solo t=0 del block del canal realiza estas escrituras; no Adam.

Conservar mu/invstd para el backward del mismo lote. La varianza no sustituye el
epsilon de Adam y las estadísticas móviles no sustituyen las del lote durante
entrenamiento. Las cuatro BN tienen sus propios parámetros y estado.

### K14 — Aplicación forward

G1D(LC): c=i%C, xhat=(U[i]-mu[c])×invstd[c];
Y[i]=gamma[c]×xhat+beta[c]. Salida distinta de U. Recalcular xhat en backward;
no se conserva un tensor adicional xhat ni se mueve Y a CPU entre capas.

### K15 — Backward, tres lanzamientos

Sea D gradiente respecto a Y. R(L,C) obtiene dos parciales por canal:
S1=sum D y S2=sum(D×xhat), usando dos shared[8,32]float32 (**2048 bytes**).
RF(P,C) reúne ambos, escribe dbeta=S1, dgamma=S2 y conserva sus valores por canal.
G1D(LC) escribe:

`dU[i] = (gamma[c] * invstd[c] / L) * (L*D[i] - S1[c] - xhat[i]*S2[c])`.

No promediar dgamma/dbeta por L ni actualizar gamma/beta aquí. Las reducciones
usan el gradiente de pérdida ya dividido por b. No recalcular estadísticas del
lote a partir de moving_mean/var. En validación CPU se usan únicamente las móviles,
sin actualizar ninguna estadística ni producir gradientes.

## H. Softmax, pérdida y gradiente: K16 (OP-P07)

Entradas Z[b,29], labels[b]. Grid=(b,1,1), block=(32,1,1), una muestra por block.
Lane t<29 carga su logit; t≥29 carga -infinito para máximo y0 para sumas.
Se usa shared float32[32] (128 bytes), reutilizado por fases con barreras.

1. Árbol de máximo strides16..1; todas las lanes leen a=max_c Zc tras barrera.
2. Calcular exp(Zc-a), suma por árbol con lanes neutrales. Cada lane válida
   obtiene pc=exp(Zc-a)/sum. Conservar P[b,29] para auditoría/contrato.
3. Lane0 escribe loss[n]=(a-Z[n,label])+log(sumexp); forma equivalente a
   logsumexp-Zlabel que evita restar números grandes al final. Etiqueta válida
   se comprueba en CPU antes de transferir y en el guard del kernel.
4. Cada lane válida escribe dZ[n,c]=(pc-indicador(c=label))/b. Se integra la
   derivada softmax+crossentropy: **no aplicar otra Jacobiana softmax**.
5. Lane0 determina la clase máxima en orden0..28 con comparación estricta,
   desempate menor índice, y escribe correcto[n]0/1. Argmax de logits equivale
   a argmax de softmax para valores finitos; no necesita D2H de probabilidades.

No se recorta pc antes de la pérdida ni se calcula log(0). Cualquier NaN/Inf
activa el flag; no se continúa entrenamiento con probabilidades inválidas.

## I. Adam: K17 (OP-P08)

Se consideran por separado W/bias de cada Conv y Dense y gamma/beta BN;
excluye medias/varianzas móviles. Todos los gradientes ya están calculados.
G1D(E) por tensor; un thread es dueño del parámetro y de sus m/v.

Para paso global t=1..1440, con escalares FP32 idénticos en CPU/GPU:

- alpha=lr×sqrt(1-beta2^t)/(1-beta1^t).
- m←m+(g-m)×(1-beta1).
- v←v+(g²-v)×(1-beta2).
- theta←theta-(m×alpha)/(sqrt(v)+epsilon), epsilon=1e-7.

Es la ubicación de epsilon del Adam Keras local; no sustituir por una fórmula
con epsilon dentro de sqrt ni por epsilon añadido a sqrt(v_hat) tras corregir
momentos: cambiaría el algoritmo. Las potencias/alpha se calculan una vez por
paso en CPU FP32 y se comparten con ambos recorridos; el coste queda incluido.
El adaptador CPU debe usar los mismos escalares y fórmula, sin ejecutar también
un optimizador estándar que duplique la actualización.

Para reproducir el redondeo de las constantes del fuente Keras, congelar
`beta1=float32(0.9)`, `beta2=float32(0.999)`, pero los coeficientes de actualización
son `float32(1.0-0.9)` y `float32(1.0-0.999)` calculados primero como constantes
Python. No sustituirlos por restas FP32 de los beta ya redondeados: no dan
necesariamente el mismo bit. La misma regla aplica a momentum BN y su complemento.
La contracción FMA que realice el compilador se registra; fastmath=False no
promete coincidencia bit a bit con CPU y las tolerancias siguen siendo obligatorias.

Tras finalizar K17 de todos los tensores, un control de un solo thread incrementa
t. Si el flag sticky de error ya está activo, K17 no modifica theta/m/v. El
coordinador comprueba t al terminar y la auditoría confirma1440, no1440×tensores.
Los flags se agregan con atomic.max **entero**, nunca atomicAdd flotante.

## J. Métricas y control: K18 y K19

### K18 — Métricas de entrenamiento

Primer pase: block256, grid1, cada lane t<b carga loss[t] y correcto[t];
resto0. Dos árboles compartidos, float32 para suma de pérdidas y int32 para
aciertos (2048 bytes en total). Segundo pase block1,grid1 añade suma al acumulador
FP32 de época, aciertos al int64 y b al contador int64. Los acumuladores se
ponen a0 antes de cada época. El promedio final es suma_loss/12180 y
correctos/12180, **sin promediar medias de lotes de distinto tamaño**.

### K19 — Control y finitud

Un buffer global int32 sticky comienza0 tras restaurar. Todo kernel que produce
valores flotantes comprueba que sus salidas sean finitas y marca error entero
si falla; solo el control del RUN puede reiniciarlo fuera del intervalo tras un
fallo. K17 comprueba también el flag antes de escribir estados. Al final de cada
paso se copia el flag4 bytes a Host y se sincroniza el stream; abortar si falla.
El paso fallido nunca se cuenta como observación correcta, aunque BN ya haya
cambiado durante su forward. Guardar el estado parcial como diagnóstico.

K19 también identifica resets de contadores/métricas (G1D pequeño) e incremento
global de paso (block1,grid1). No inicializar gradientes a0 para luego usar +=
sin necesidad: cada productor los escribe por completo. No son operaciones de
aprendizaje adicionales ni una forma de ocultar sincronización en el cronometraje.

## K. Ejemplos concretos y cobertura

Ejemplos **calculados**, b=128; configuración lógica seguida de física cuando
requiere slabs. El número de blocks no es el número de blocks residentes.

| Operación | M,N,K / tamaño | Block | Grid y lanzamientos |
|---|---|---|---|
| C1 forward | 524288,32,27 | 16×16 | Lógico(2,32768,1); 32 lanzamientos(2,1024,1) |
| C2 forward | 524288,32,288 | 16×16 | Igual grid a C1, distinta reducción |
| C3 forward | 131072,64,288 | 16×16 | Lógico(4,8192,1); 8 lanzamientos(4,1024,1) |
| C5 forward | 32768,128,576 | 16×16 | Lógico(8,2048,1); 2 lanzamientos(8,1024,1) |
| C2 dW | M288,N32,K524288; P256 | 16×16 | (2,18,256), un lanzamiento y posterior K04 |
| F1 forward | 128,256,8192 | 16×16 | (16,8,1) |
| F1 dW | 8192,256,128; P1 | 16×16 | (16,512,1) |
| B1 primera reducción | L524288,C32,P256 | 32×8 | (256,1,1); final block256,grid(32,1,1) |
| B4 primera reducción | L128,C256,P1 | 32×8 | (1,8,1); final block256,grid(256,1,1) |
| P1 forward | 4194304 elementos salida | 256 | (4096,1,1), cuatro iteraciones por thread |
| Softmax/pérdida | 128 muestras,29 clases | 32 | (128,1,1) |
| C1 forward, último lote20 | M81920,N32,K27 | 16×16 | 5 lanzamientos(2,1024,1), sin leer108 imágenes de cola |
| F1 forward, último lote20 | M20,N256,K8192 | 16×16 | (16,2,1), últimas12 filas neutras |

| Oportunidad paso3 | Kernels / diseño que la cubren |
|---|---|
| OP-P01 convoluciones | K01 |
| OP-P02 Dense | K05 |
| OP-P03 gradientes | K02/03/04/06/07/08 y backward de cada capa |
| OP-P04 ReLU | Fusión K01/K05 y K09 |
| OP-P05 pooling | K11/12 |
| OP-P06 BN | K13/14/15 |
| OP-P07 softmax/pérdida | K16/18 |
| OP-P08 Adam | K17/19 |
| OP-P09 Dropout | K10 y generación de máscaras CPU compartidas |

No se usa una barrera global dentro de un kernel, no hay retorno temprano que
salte barreras, ni escritura de dos threads al mismo gradiente. Los peligros de
alias, últimas dimensiones, padding, máscaras, desempates y división por b real
tienen pruebas obligatorias en el [plan de validación](diseno-validacion-cuda.md).
