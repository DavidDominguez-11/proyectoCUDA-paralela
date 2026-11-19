/* Native editable project presentation. Reads saved results only; no training. */
const fs=require('fs'),path=require('path');
const ROOT=path.resolve(__dirname,'..');
const pptxgen=require(path.join(ROOT,'data/presentacion-tooling/node_modules/pptxgenjs'));
const JSZip=require(path.join(ROOT,'data/presentacion-tooling/node_modules/jszip'));
const D=JSON.parse(fs.readFileSync(path.join(ROOT,'entregables/presentacion/datos-presentacion.json'),'utf8'));
const pres=new pptxgen();pres.layout='LAYOUT_WIDE';
pres.author='David Domínguez · Jonathan Diaz · Luis Padilla';pres.subject='Proyecto 3: Mejorando algoritmos con CUDA';
pres.title='De CPU a CUDA: reconocimiento ASL';pres.company='Universidad del Valle de Guatemala';pres.lang='es-GT';
const THEME={name:'ASL CUDA / Execution tiles',headFontFace:'Calibri',bodyFontFace:'Calibri',colors:{
 dk1:'15222D',lt1:'FFFFFF',dk2:'536775',lt2:'EDF2F5',accent1:'009E73',accent2:'0072B2',accent3:'D55E00',accent4:'9162B0',accent5:'56B4E9',accent6:'C7D3DA',hlink:'0072B2',folHlink:'9162B0'}};
pres.theme={headFontFace:THEME.headFontFace,bodyFontFace:THEME.bodyFontFace,lang:'es-GT'};
const C=pres.SchemeColor,S=pres.ShapeType,CPU=THEME.colors.accent2,GPU=THEME.colors.accent3;
const outline=[],bounds=[];let section='',number=0;
for(const dark of [false,true]){
 pres.defineSlideMaster({title:dark?'DARK':'LIGHT',background:{color:dark?C.text1:C.background1},margin:.6,
  objects:[{placeholder:{text:'',options:{name:'title',type:'title',x:.6,y:.43,w:12.1,h:.75,fontSize:36,bold:true,color:dark?C.background1:C.text1,margin:0,breakLine:false}}},
   {text:{text:'ASL  /  PROYECTO 3 CUDA',options:{x:.6,y:7.08,w:7,h:.18,fontSize:10,color:dark?C.accent6:C.text2,margin:0,isTextBox:true,objectName:'Pie de página'}}}],
  slideNumber:{x:12.08,y:7.04,w:.65,h:.25,fontSize:10,color:dark?C.accent6:C.text2,align:'right',margin:0}});
}
function sec(name){section=name;pres.addSection({title:name});}
function slide(title,notes,source='',dark=false){
 const s=pres.addSlide({masterName:dark?'DARK':'LIGHT',sectionTitle:section});number++;
 s.addText(title,{placeholder:'title'});
 s.addNotes(notes+'\n\nProcedencia / respaldo: '+source+'\nLas cifras son mediciones guardadas o cálculos derivados; no se entrenó para crear esta presentación.');
 outline.push({numero:number,seccion:section,titulo:title,notas:notes,fuentes:source});return s;
}
function text(s,t,x,y,w,h=0.4,size=18,color=C.text1,bold=false,extra={}){
 s.addText(String(t),{x,y,w,h,fontSize:size,color,bold,margin:0,breakLine:false,isTextBox:true,objectName:String(t).slice(0,70),...extra});
 bounds.push({slide:number,type:'text',text:String(t),x,y,w,h});
}
function box(s,x,y,w,h,fill=C.background2,line=fill,r=.15){s.addShape(S.roundRect,{x,y,w,h,rectRadius:r,radius:r,fill:{color:fill},line:{color:line,width:0},objectName:'Bloque de ejecución'});}
function line(s,x,y,w,h,color=C.text2,arrow=true){s.addShape(S.line,{x,y,w,h,line:{color,width:1.6,...(arrow?{endArrowType:'triangle'}:{})},objectName:'Flujo de datos'});}
function note(s,t,x=.65,y=6.62,w=12,size=12){text(s,t,x,y,w,.32,size,C.text2);}
function badge(s,label,x,y,w=1.7,fill=C.background2,col=C.text1){box(s,x,y,w,.4,fill);text(s,label,x+.12,y+.07,w-.24,.23,12,col,true);}
function stat(s,value,label,x,y,w=3.6,sub='',color=C.accent1,dark=false){
 text(s,value,x,y,w,.8,48,color,true);text(s,label,x,y+.88,w,.53,19,dark?C.background1:C.text1,true);
 if(sub)text(s,sub,x,y+1.48,w,.62,14,dark?C.accent6:C.text2);
}
function table(s,headers,rows,x=.65,y=1.65,w=12.03,cols=null,size=16,rowH=.48){
 const opts={x,y,w,fontSize:size,color:C.text1,border:{type:'solid',pt:.5,color:THEME.colors.accent6},
  fill:C.background1,margin:[.07,.11,.07,.11],rowH,autoPage:false,breakLine:false,
  showBorder:true,verbose:false,objectName:'Tabla editable de resultados'};
 if(cols)opts.colW=cols;
 const hdr=headers.map(h=>({text:h,options:{bold:true,color:C.background1,fill:C.text1}}));
 const data=rows.map((r,i)=>r.map(v=>({text:String(v),options:{fill:i%2?C.background2:C.background1}})));
 s.addTable([hdr,...data],opts);
}
function picture(s,key,x,y,w,h,label){
 const im=D.images[key],ratio=Math.min(w/im.w,h/im.h),iw=im.w*ratio,ih=im.h*ratio;
 s.addImage({path:path.join(ROOT,im.path),x:x+(w-iw)/2,y:y+(h-ih)/2,w:iw,h:ih,altText:label||key,objectName:'Evidencia original: '+(label||key)});
}
function chart(s,type,data,x,y,w,h,extra={}){
 s.addChart(type,data,{x,y,w,h,showTitle:false,showValue:false,showLegend:false,showBorder:false,
  showCatName:false,showMarker:type===pres.ChartType.line,markerSize:5,lineSize:2.5,
  chartColors:[CPU,GPU,THEME.colors.accent1,THEME.colors.accent4,THEME.colors.accent5],
  catAxisLabelFontFace:'+mn-lt',valAxisLabelFontFace:'+mn-lt',legendFontFace:'+mn-lt',dataLabelFontFace:'+mn-lt',dataLabelFormatCode:'0.0',
  catAxisLabelFontSize:14,valAxisLabelFontSize:12,legendFontSize:13,dataLabelBkgrdColor:THEME.colors.lt1,
  catAxisLabelColor:THEME.colors.dk2,valAxisLabelColor:THEME.colors.dk2,
  valGridLine:{color:THEME.colors.lt2,width:.5},catGridLine:{style:'none'},
  showCatName:false,showSerName:false,showCatName:false,showCatName:false,
  showValue:false,showCatName:false,showPercent:false,
  catAxisLineShow:false,valAxisLineShow:false,
  showLegend:false,showTitle:false,showBorder:false,
  objectName:'Gráfica nativa editable',...extra});
 bounds.push({slide:number,type:'chart',x,y,w,h});
}
function row(name,phase='entrenamiento_'){return D.compare.find(r=>r.fase===phase+name);}
function n(v,d=3){return Number(v).toFixed(d);}
function fitobs(model,origin){return D.observations.filter(r=>r.modelo===model&&r.origen===origin&&r.fase==='entrenamiento_'+model).sort((a,b)=>+a.repeticion-+b.repeticion).map(r=>+r.duracion_s);}
function cv(a){const m=a.reduce((x,y)=>x+y,0)/a.length;return 100*Math.sqrt(a.reduce((x,y)=>x+(y-m)**2,0)/(a.length-1))/m;}
function blocks(s,x,y,cols=8,rows=4,color=C.accent1,size=.18,gap=.055){for(let j=0;j<rows;j++)for(let i=0;i<cols;i++)s.addShape(S.rect,{x:x+i*(size+gap),y:y+j*(size+gap),w:size,h:size,fill:{color,transparency:(i+j)%4===0?45:0},line:{color,width:0},objectName:'Hilo CUDA'});}
function step(s,num,title,sub,x,y,w=2.25,color=C.accent1){box(s,x,y,w,1.13);text(s,String(num).padStart(2,'0'),x+.16,y+.12,.48,.4,25,color,true);text(s,title,x+.72,y+.14,w-.85,.4,18,C.text1,true);text(s,sub,x+.17,y+.68,w-.3,.3,13,C.text2);}

sec('Inicio y solución secuencial');
{
 const s=slide('De CPU a CUDA','Presentar el reconocimiento de señas ASL como caso de aplicación. El proyecto parte del laboratorio original, mide una referencia CPU y traslada el entrenamiento completo de CNN-B, FC-A y FC-B a CUDA Python. Los tres números son speedup de entrenamiento total medido, con transferencias y validación CPU incluidas. No son aceleración de todo el notebook.','Comparación ampliación FC, n=5 por modelo; guía Proyecto 3 CUDA',true);
 text(s,'Reconocimiento ASL · del archivo a la predicción',.65,1.4,10,.5,25,C.accent6);
 for(let i=0;i<3;i++){picture(s,['A','B','C'][i],.65+i*1.1,2.28,.92,.92);}
 blocks(s,8.6,1.75,13,7,C.accent1,.19,.07);
 text(s,'3 redes entrenadas en GPU',.65,3.6,10,.6,30,C.background1,true);
 ['CNN-B','FC-A','FC-B'].forEach((m,i)=>stat(s,n(row(m).speedup,2)+'×',m,.65+i*4.05,4.43,3.7,'CPU / CUDA · media de 5 pruebas',C.accent1,true));
 text(s,'David Domínguez 23712  ·  Jonathan Diaz 23873  ·  Luis Padilla 23663',.65,6.65,11.7,.25,13,C.accent6);
}
{
 const s=slide('El proyecto en seis pasos','Seguir el mismo orden que la guía. Los diagramas resumen el procedimiento; después se concentra la exposición en tiempos, variabilidad, calidad y evidencia de recursos. La extensión FC está integrada al diseño y a las conclusiones finales.','Guía principal, apartados 1–6');
 const a=[['Analizar','Flujo y operaciones'],['Medir CPU','5 corridas equivalentes'],['Seleccionar','Matemática repetitiva'],['Diseñar','Threads · blocks · grids'],['Implementar','CUDA Python y cuBLAS'],['Comparar','Tiempo · speedup · calidad']];
 a.forEach((q,i)=>step(s,i+1,...q,.7+(i%3)*4.1,1.75+Math.floor(i/3)*2.1,3.75));
 note(s,'Misma carga de trabajo → mediciones conservadas → resultados verificables',.7,6.35,12,16);
}
{
 const s=slide('De 87,000 imágenes a una prueba controlada','El dataset ASL Alphabet contiene 29 clases, letras A–Z y las clases del, nothing y space. Se seleccionaron 600 imágenes por clase, 17,400 en total, con particiones estratificadas 70/15/15. Las seis fotografías provienen de archivos reales del manifiesto de entrenamiento; no son imágenes generadas. Se trabajó con archivos almacenados, no con video en vivo.','Manifiesto EXP-002; selección y hashes de seis archivos en datos-presentacion.json');
 ['A','B','C','M','N','S'].forEach((a,i)=>{picture(s,a,.7+i*2.05,1.6,1.75,1.75);text(s,a,.7+i*2.05,3.43,1.75,.3,18,C.text1,true,{align:'center'});});
 stat(s,'29','clases',.7,4.0,2.8,'600 imágenes por clase');stat(s,'17,400','imágenes seleccionadas',4.65,4,3.8,'RGB · 64×64 · float32');
 box(s,9.3,4,3.25,1.7);text(s,'/255 → [0,1]',9.5,4.2,2.85,.5,25,C.accent1,true);text(s,'Mismos tensores CPU / GPU',9.5,4.92,2.85,.46,15,C.text2);
 chart(s,pres.ChartType.bar,[{name:'Train',labels:['Partición'],values:[70]},{name:'Validación',labels:['Partición'],values:[15]},{name:'Test',labels:['Partición'],values:[15]}],.7,5.78,11.9,.7,{barDir:'bar',catAxisHidden:true,valAxisHidden:true,showLegend:false,catName:'',barGrouping:'stacked',chartColors:[THEME.colors.accent1,THEME.colors.accent2,THEME.colors.accent3],showValue:true,dataLabelPosition:'ctr',dataLabelColor:'FFFFFF',dataLabelFormatCode:'0"%"'});
 note(s,'Train 12,180   ·   Validación 2,610   ·   Test 2,610',.85,6.55,11.8,16);
}
{
 const s=slide('1 · ¿Qué hacía el programa original?','Describir la entrada de imágenes, la selección y transformación a tensores. El laboratorio entrena cuatro redes base y una variante con aumento, además de PCA y Random Forest. Después compara, guarda e infiere sobre fotos del equipo. La rama aumentada cambia según la selección: por eso su total adaptativo no se usó como denominador de speedup. El original se conserva intacto.','Notebook original y análisis original; CPU EXP-002');
 const flow=[['Archivos JPEG','Dataset local'],['Preparación','RGB → 64×64 → /255'],['Particiones','Train / val / test']];
 flow.forEach((q,i)=>{step(s,i+1,...q,.7+i*4.08,1.6,3.7);if(i<2)line(s,4.45+i*4.08,2.12,.25,0);});
 line(s,10.72,2.95,0,.55);
 const models=[['CNN-A / CNN-B','Convoluciones'],['FC-A / FC-B','Capas densas'],['PCA + RF','Modelos clásicos']];
 models.forEach((q,i)=>{box(s,.7+i*4.08,3.7,3.7,1.0);text(s,q[0],.9+i*4.08,3.9,3.3,.34,21,C.text1,true);text(s,q[1],.9+i*4.08,4.32,3.3,.25,14,C.text2);});
 line(s,6.48,4.9,0,.45);
 box(s,1.45,5.58,10.4,.7,C.text1);text(s,'Evaluar → comparar → aumento de imágenes → guardar → predecir',1.65,5.78,10.0,.33,20,C.background1,true,{align:'center'});
 note(s,'La versión original también usa paralelismo interno de librerías CPU; “secuencial” identifica la referencia');
}
{
 const s=slide('1 · Las operaciones que se repiten','Explicar cómo una imagen termina generando miles de operaciones por capa. Se cargan 17,400 imágenes; cada red procesa 12,180 por época, en 96 lotes, incluyendo uno de 20. Quince épocas son 1,440 actualizaciones Adam y 182,700 presentaciones de imágenes de entrenamiento por modelo. Las 15 validaciones procesan 39,150 imágenes y no actualizan pesos. Estos son los volúmenes de la campaña fija, no el EarlyStopping histórico del notebook original.','Análisis original; configuraciones EXP-002/003/012');
 table(s,['Operación identificada','Datos que procesa','Repeticiones por modelo'],[
 ['Abrir / transformar archivos','17,400 imágenes','1 por imagen preparada'],
 ['Recorrer datos de entrenamiento','12,180 imágenes','15 épocas'],['Formar lotes','Batch 128; último 20','96 lotes / época'],
 ['Forward + pérdida + backward','Activaciones y pesos','1,440 lotes'],['Actualizar Adam','Pesos + momentos + velocidades','1,440 actualizaciones'],
 ['Validar en CPU','2,610 imágenes','15 pasadas'],['Evaluar al final','2,610 imágenes test','1 pasada']
 ],.7,1.65,11.9,[4.0,4.0,3.9],17,.55);
 badge(s,'182,700 imágenes vistas durante entrenamiento',.7,6.35,8.0,C.background2,C.accent1);
}
{
 const s=slide('2 · Referencia CPU: medir antes de cambiar','La referencia no es estrictamente monohilo. TensorFlow CPU usa seis hilos intra-op y uno inter-op, BLAS seis; Random Forest mantiene n_jobs=-1. El calentamiento y las importaciones quedan fuera; se restaura el estado antes de medir. Las cinco corridas completas válidas son CPU01–05. La carga base fija comparte 14 fases y tiene variación baja. El total adaptativo mezcla distintas CNN aumentadas y no es una referencia homogénea para CUDA.','EXP-002: serie.json, resultados.md y resumen_referencia_comun.csv');
 step(s,1,'Verificar','Datos + estado inicial',.7,1.65,3.75);step(s,2,'Calentar','Restaurar después',4.8,1.65,3.75);step(s,3,'Cronometrar','perf_counter_ns',8.9,1.65,3.75);
 table(s,['Corrida','14 fases comunes (s)'],D.commonruns.map((v,i)=>['CPU '+(i+1),n(v)]),.75,3.25,5.4,[2.2,3.2],17,.43);
 stat(s,n(D.common.media_s,2)+' s','media de la referencia común',6.95,3.18,5.0,'DE 2.72 s  ·  CV 0.24%  ·  n=5');
 badge(s,'15 épocas fijas · batch 128 · FP32',6.95,5.45,5.05);note(s,'El total adaptativo (1,858.62 s) se conserva aparte: no se utiliza para calcular speedup',.75,6.55,12,14);
}
{
 const s=slide('2 · CNN-B concentraba el tiempo CPU','Se suman las medias de 14 fases no solapadas. CNN-B representa 71.94% del total común. FC-A y FC-B aportan más tiempo de entrenamiento paralelizable; con las tres se cubre aproximadamente 85.61% de esta referencia. El donut no representa el pipeline adaptativo completo. La validación dentro de fit ya está incluida y no se añade otra vez.','CPU EXP-002 resumen.csv; referencia común 1119.9275623 s');
 const b=(p)=>+D.baseline.find(r=>r.fase===p).media_s,total=+D.common.media_s;
 const v=[b('entrenamiento_CNN-B'),b('entrenamiento_CNN-A'),b('entrenamiento_FC-B'),b('entrenamiento_FC-A')];v.push(total-v.reduce((a,b)=>a+b,0));
 chart(s,pres.ChartType.doughnut,[{name:'Tiempo CPU',labels:['CNN-B','CNN-A','FC-B','FC-A','Otras fases'],values:v}],.65,1.65,6.15,4.8,{holeSize:72,showLegend:true,legendPos:'b',chartColors:[GPU,'C7D3DA',THEME.colors.accent1,THEME.colors.accent2,'E5EBEF'],showPercent:false});
 text(s,'71.94%',2.13,3.15,3.15,.8,44,C.accent3,true,{align:'center'});text(s,'CNN-B',2.6,4.0,2.2,.4,20,C.text2,true,{align:'center'});
 table(s,['Entrenamiento','Tiempo CPU (s)','Parte del total'],['CNN-B','FC-B','FC-A','CNN-A'].map(m=>[m,n(b('entrenamiento_'+m),2),n(100*b('entrenamiento_'+m)/total,2)+'%']),7.1,2,5.5,[1.9,1.9,1.7],16,.68);
 text(s,'Prioridad inicial: CNN-B\nAmpliación final: FC-A y FC-B',7.1,5.52,5.5,.82,18,C.text2);
}
sec('Selección, diseño e implementación');
{
 const s=slide('3 · ¿Qué se paralelizó y por qué?','No se trasladó todo el programa al GPU. Se seleccionaron la matemática del entrenamiento de tres redes y sus operaciones por elemento o por reducción. La carga de archivos queda en CPU porque implica rutas, sistema de archivos, decodificación e I/O irregular. PCA, Random Forest, CNN-A y augmentation no se portaron en este alcance. La validación y test siguen en CPU por el diseño aplicado; sus costes se declaran.','Paso3, implementación revisada y ampliación FC');
 table(s,['Operación','¿Qué puede ocurrir a la vez?','Ejecutor final'],[
 ['Archivos / decodificación / resize','Imágenes independientes','CPU · 6 workers'],['Convoluciones / Dense','Productos y sumas de matrices','GPU · cuBLAS'],
 ['ReLU / Dropout / normalización','Elementos y características del lote','GPU · kernels Python'],['Pérdida / gradientes / Adam','Muestras, parámetros y reducciones','GPU · kernels Python'],
 ['Validación / test','Predicciones del modelo exportado','CPU'],['PCA / RF / CNN-A / augmentation','Fuera del alcance CUDA elegido','CPU original']
 ],.7,1.7,11.95,[4.1,4.65,3.2],17,.63);
 badge(s,'CNN-B + FC-A + FC-B: entrenamiento completo en GPU',.7,6.25,10.9,C.background2,C.accent1);
}
{
 const s=slide('4 · Un flujo CPU ↔ GPU explícito','Recorrer la separación de fases. CPU realiza lectura y preparación, genera el orden de lotes y máscaras y usa memoria host fijada para copiar. H2D envía datos al stream. GPU hace forward, pérdida, backward y Adam. Al final de cada época, D2H exporta pesos y estadísticas BN para validar en CPU. El ciclo se repite quince veces. Preparación no se presenta como CUDA y test no se presenta como inferencia GPU.','Diseño aplicado: modelo Network/FCNetwork, un stream, sin solapamiento');
 box(s,.7,1.8,3.3,3.65);text(s,'CPU / HOST',.95,2.05,2.8,.5,25,C.accent2,true);
 ['Archivos y decodificación','RGB / resize / normalización','6 workers','Lotes y máscaras Dropout'].forEach((a,i)=>text(s,a,.95,2.82+i*.52,2.9,.38,17,C.text1,i===2));
 box(s,5.15,1.8,3.2,3.65,C.text1);text(s,'GPU / DEVICE',5.4,2.05,2.7,.5,24,C.background1,true);
 ['Forward + pérdida','Backward / gradientes','Adam + métricas'].forEach((a,i)=>text(s,a,5.4,2.9+i*.57,2.7,.4,18,C.background1,true));blocks(s,5.45,4.75,10,2,C.accent1,.15,.05);
 box(s,9.6,1.8,3.0,3.65);text(s,'CPU / EVALUAR',9.85,2.05,2.55,.5,23,C.accent2,true);
 ['Validación cada época','Test final','Métricas / guardado'].forEach((a,i)=>text(s,a,9.85,2.92+i*.57,2.55,.42,17,C.text1));
 line(s,4.1,3.15,.9,0,C.accent3);text(s,'H2D',4.12,2.7,.85,.3,15,C.accent3,true,{align:'center'});
 line(s,8.48,3.15,.96,0,C.accent3);text(s,'D2H',8.5,2.7,.85,.3,15,C.accent3,true,{align:'center'});
 note(s,'Por época: 96 lotes → exportar → validar en CPU → continuar hasta 15 épocas',.9,5.95,11.4,20);
}
{
 const s=slide('4 · Tres redes, la misma regla de reparto','FC significa fully connected: red de capas densas, diferente de la CNN que aprende filtros espaciales. CNN-B contiene cinco convoluciones con 32/32/64/64/128 filtros, tres pooling y Dense256. Las FC aplanan los 12,288 valores de la imagen y usan BN al inicio. Se preservan las arquitecturas, incluyendo Dropout y la diferencia de momentum BN. Las cifras son parámetros entrenables, no el número de operaciones ni la VRAM observada.','Construcciones originales + FCNetwork; inicializaciones históricas preservadas');
 const a=[{name:'CNN-B',layers:['64×64×3','Conv 32 → Conv 32','Pool → Conv 64 → Conv 64','Pool → Conv 128 → Pool','Flatten → Dense 256','Softmax 29'],params:'2,245,245',extra:'5 Conv · 3 Pool · BN momentum 0.9'},
 {name:'FC-A',layers:['64×64×3','Flatten 12,288','BatchNormalization','Dense 128 → Dense 64','Dropout 0.3 por capa oculta','Softmax 29'],params:'1,607,709',extra:'BN de entrada · momentum 0.99'},
 {name:'FC-B',layers:['64×64×3','Flatten 12,288','BatchNormalization','Dense 512 → 256 → 128','Dropout 0.4 por capa oculta','Softmax 29'],params:'6,484,509',extra:'BN de entrada · momentum 0.99'}];
 a.forEach((q,i)=>{let x=.7+i*4.08;box(s,x,1.62,3.7,4.68);text(s,q.name,x+.2,1.82,3.3,.5,27,C.accent1,true);
  q.layers.forEach((r,j)=>{box(s,x+.2,2.55+j*.43,3.3,.34,j===5?C.text1:C.background1);text(s,r,x+.32,2.61+j*.43,3.06,.24,14,j===5?C.background1:C.text1,j===5);});
  text(s,q.params,x+.2,5.4,3.3,.4,24,C.text1,true);text(s,'parámetros entrenables',x+.2,5.87,3.3,.25,13,C.text2);note(s,q.extra,x,6.4,3.9,11.2);});
}
{
 const s=slide('4 · Threads, blocks y grids','Explicar los tres niveles CUDA con el dibujo. Los kernels propios distribuyen elementos en hilos de bloques de 256 y usan recorrido grid-stride. La reducción parcial BN usa 32×8; la final FC asigna un hilo por característica. Softmax usa 32 hilos y un bloque por muestra. Los productos SGEMM se delegan a cuBLAS: su reparto interno es de NVIDIA y no se atribuye al equipo.','Código CUDA medido; recursos compilados y lanzamientos por RUN');
 blocks(s,.85,1.65,8,4,C.accent1,.18,.065);text(s,'THREADS',.85,2.88,2.5,.35,20,C.accent1,true);text(s,'elementos independientes',.85,3.34,3,.4,15,C.text2);
 for(let j=0;j<2;j++)for(let i=0;i<3;i++){box(s,4.55+i*.58,1.65+j*.58,.45,.45,C.accent2);}
 text(s,'BLOCKS',4.55,2.88,2.6,.35,20,C.accent2,true);text(s,'hilos agrupados',4.55,3.34,3,.4,15,C.text2);
 for(let j=0;j<3;j++)for(let i=0;i<5;i++){box(s,8.6+i*.45,1.65+j*.35,.3,.22,C.accent3);}
 text(s,'GRID',8.6,2.88,2.6,.35,20,C.accent3,true);text(s,'todos los bloques del kernel',8.6,3.34,3.8,.4,15,C.text2);
 table(s,['Operación','Block','Grid / asignación'],[
 ['ReLU / Dropout / Adam','256 hilos','ceil(E / 256), tope 4,096'],['BN parcial FC, C=12,288','32 × 8 hilos','1 × 384 bloques'],
 ['BN final FC','256 hilos','48 bloques'],['Softmax / pérdida','32 hilos','128 bloques; último lote: 20'],['Matrices Dense / Conv','cuBLAS SGEMM','Distribución interna de NVIDIA']
 ],.7,4.02,11.95,[4.7,2.6,4.65],15,.39);
}
{
 const s=slide('5 · Qué se implementó y qué se midió','La implementación usa Numba-CUDA para kernels Python y cuBLAS SGEMM para los productos grandes; TensorFlow/Keras está restringido a CPU y espejo para validar/guardar. El intervalo principal es pared sincronizada y contiene staging/máscaras, H2D, forward/backward/Adam, D2H y validación CPU. Preparación de imágenes, warmup/compilación y guardado/figuras se separan. Un único stream mantiene el orden y no hay solapamiento.','EXP-003 y colección FC EXP-011/012; fuentes congeladas y temporizadores reales');
 const libs=[['Numba-CUDA','Kernels Python explícitos'],['cuBLAS SGEMM','Matrices FP32 · modo PEDANTIC'],['TensorFlow / Keras','Solo CPU: validar y guardar']];
 libs.forEach((a,i)=>{box(s,.7+i*4.08,1.65,3.7,1.13);text(s,a[0],.9+i*4.08,1.85,3.3,.4,23,C.accent1,true);text(s,a[1],.9+i*4.08,2.4,3.3,.25,14,C.text2);});
 badge(s,'Fuera del entrenamiento medido',.7,3.26,4.45);text(s,'Warmup → restaurar → preparar imágenes',5.5,3.3,7.1,.4,19,C.text2);
 box(s,.7,4.03,11.95,1.45,C.text1);
 ['Staging / máscaras','H2D','Forward / backward / Adam','D2H','Validación CPU'].forEach((a,i)=>{const xx=[.95,3.65,4.6,8.8,9.65][i],ww=[2.35,.65,3.85,.65,2.6][i];text(s,a,xx,4.43,ww,.55,18,C.background1,true);});
 text(s,'← Tiempo pared sincronizado: todo el bloque está incluido →',.95,5.65,11.4,.4,20,C.accent3,true,{align:'center'});
 note(s,'Después: test CPU cronometrado aparte; guardado y gráficas fuera de fit',.7,6.3,12,15);
}
{
 const s=slide('5 · Verificar antes de confiar en el tiempo','Las pruebas locales contrastan operadores y gradientes con un oráculo CPU. Se verificó la restauración tras warmup y el contador Adam real de los checkpoints. La corrección local aprobada no implica una trayectoria numérica idéntica durante quince épocas: RNG, orden y redondeo difieren y en la CNN hubo diagnósticos estrictos de rutas cerca de empates que no se etiquetaron como aprobados. Se conserva esa reserva.','Pruebas operadores/adjuntos CNN-B; validacion-01 FC; auditoría HDF5 final');
 stat(s,'51','pruebas de operadores CNN-B',.8,1.8,5.5,'Tolerancias locales verificadas');stat(s,'78','adjuntos / gradientes CNN-B',7.05,1.8,5.5,'Activaciones y dY compartidos');
 stat(s,'4','casos FC-A / FC-B',.8,4.03,5.5,'Lotes 20 y 128 · error probabilidades ≤1.67×10⁻⁶');stat(s,'1,440','pasos Adam por corrida',7.05,4.03,5.5,'Checkpoint real · valores finitos');
 note(s,'Pruebas numéricas locales aprobadas ≠ trayectoria completa CPU/GPU idéntica',.8,6.55,11.8,15);
}
sec('Evidencia de ejecución');
{
 const s=slide('5 · CPU real durante su fase de trabajo','Esta captura pertenece a la ejecución adicional n=1 del notebook CPU, separada de los promedios de la campaña. Fue tomada durante entrenamiento CNN-B en CPU, no durante preparación. psutil usa 100% por procesador lógico, por eso 507.3% indica varios núcleos activos. GPU estaba a0%. La evidencia de preparación con seis workers se presenta en el respaldo de recursos; no atribuir esta captura a I/O.','EXP-008 captura-cpu-sostenido.png y JSON de muestra original');
 picture(s,'cpu',.65,1.55,8.65,4.95,'Captura real CPU: fase entrenamiento CNN-B');
 stat(s,'507.3%','CPU proceso + hijos',9.6,2.0,3.0,'100% equivale a un núcleo',C.accent2);
 stat(s,'0%','GPU en esta muestra',9.6,4.4,3.0,'Captura durante CPU',C.accent2);
 note(s,'Monitor en vivo · lectura original psutil / nvidia-smi · muestra sostenida, época 3');
}
{
 const s=slide('5 · GPU real durante entrenamiento CNN-B','La captura es de la corrida adicional de notebook CUDA y no modifica las cinco medias previas. nvidia-smi registró91% de actividad GPU y1,213MiB de4,096, un29.6% de capacidad VRAM. Actividad agregada no significa91% de núcleos ni ocupación por SM. Es evidencia de uso real y actividad elevada; no prueba saturación total de memoria.','EXP-009 captura-cuda.png y metadata original');
 picture(s,'cnn',.65,1.55,8.65,4.95,'Captura real entrenamiento CNN-B CUDA');
 stat(s,'91%','actividad GPU',9.6,1.93,3.0,'Agregado nvidia-smi',C.accent3);stat(s,'1,213','MiB de VRAM',9.6,4.35,3.0,'29.6% de los 4,096 MiB',C.accent3);
 note(s,'Actividad GPU ≠ ocupación por SM; VRAM utilizada ≠ saturación de toda la memoria');
}
{
 const s=slide('5 · FC-A y FC-B también usaron CUDA','Son capturas originales durante la primera corrida válida de cada FC. FC-A registró50% de actividad y111MiB; FC-B32% y187MiB en esas muestras. Las capturas no son máximos globales. Una GPU menos activa puede terminar antes cuando tiene menos trabajo por lote o mayores proporciones de host y validación; el speedup se determina por tiempo total. No se llenó VRAM artificialmente.','EXP-011 FC-A1 y EXP-012 FC-B1: capturas y CSV originales');
 picture(s,'fcA',.65,1.55,5.85,4.3,'Captura real FC-A');picture(s,'fcB',6.85,1.55,5.85,4.3,'Captura real FC-B');
 text(s,'FC-A  ·  50% GPU  ·  111 MiB',.85,5.98,5.5,.45,22,C.accent3,true);text(s,'FC-B  ·  32% GPU  ·  187 MiB',7.05,5.98,5.5,.45,22,C.accent3,true);
 note(s,'Capturas puntuales reales; los promedios y máximos del monitor están en el anexo');
}
sec('Resultados y conclusiones');
{
 const s=slide('6 · Tres redes, menos tiempo total','Esta es la tabla principal de resultados. Cada cifra resume cinco corridas por dispositivo/modelo. CUDA incluye transferencias y validación CPU. Los tres minigráficos tienen escalas verticales diferentes: comparan CPU/GPU dentro de cada arquitectura. Mostrar la fórmula speedup como cociente entre medias, no media de cocientes y no suma de speedups. La CPU es histórica; no se repitió la campaña para preparar la ampliación.','comparacion/ampliacion-fc/comparacion.csv y estadisticas.csv');
 ['CNN-B','FC-A','FC-B'].forEach((m,i)=>{const r=row(m),xx=.7+i*4.08;text(s,m,xx,1.52,3.7,.4,22,C.text1,true);
  chart(s,pres.ChartType.bar,[{name:'Tiempo',labels:['CPU','CUDA'],values:[+r.cpu_media_s,+r.cuda_media_s]}],xx,2.0,3.7,2.9,{barDir:'col',showValue:true,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.0" s"',chartColors:[CPU,GPU],showLegend:false,showCatName:false,valAxisMinVal:0,valAxisMaxVal:Math.ceil(+r.cpu_media_s*1.2/10)*10,catAxisLabelFontSize:16});
 });
 table(s,['Entrenamiento','CPU media ± DE (s)','CUDA media ± DE (s)','Speedup','Menos tiempo'],['CNN-B','FC-A','FC-B'].map(m=>{const r=row(m);return[m,n(r.cpu_media_s,2)+' ± '+n(r.cpu_de_s,2),n(r.cuda_media_s,2)+' ± '+n(r.cuda_de_s,2),n(r.speedup,2)+'×',n(r.reduccion_pct,2)+'%'];}),.7,5.12,11.95,[2.25,3.0,3.0,1.8,1.9],16,.43);
 note(s,'Escalas distintas por panel · Speedup = media CPU / media CUDA · n=5 · DE muestral',.7,6.7,12,12);
}
{
 const s=slide('6 · Las cinco repeticiones, sin escoger la mejor','Mostrar todas las observaciones de entrenamiento. No se eligió solo la ejecución más rápida. FC-A1 guardada en EXP-011 se reutilizó después de corregir solo la presentación; EXP-012 hizo las nueve pendientes. Un intento FC-A incompleto de EXP-010 está excluido del promedio por no cerrar/checkpointear, no por su velocidad. Las cinco repeticiones CUDA parten del mismo estado y RNG: la variación resume tiempo, no cinco aprendizajes independientes.','observaciones.csv; series CPU01–05, GPU01–05, FC colección EXP-012');
 const names=['CNN-B','FC-A','FC-B'];
 table(s,['RUN','CNN-B CPU','CNN-B CUDA','FC-A CPU','FC-A CUDA','FC-B CPU','FC-B CUDA'],[0,1,2,3,4].map(i=>[String(i+1),...names.flatMap(m=>[n(fitobs(m,'CPU')[i],3),n(fitobs(m,'CUDA')[i],3)])]),.65,1.8,12.03,[.65,1.9,1.93,1.8,1.9,1.88,1.97],15.5,.46);
 names.forEach((m,i)=>stat(s,n(cv(fitobs(m,'CUDA')),2)+'%',m+' · CV CUDA',.75+i*4.08,4.98,3.8,'',C.accent1));
 note(s,'Tiempos en segundos · 15 épocas por red y corrida · sin mezclar las observaciones adicionales n=1',.7,6.7,12,12);
}
{
 const s=slide('6 · Comparar fases sin mezclar causas','La preparación mejoró de17.325s a8.034s con seis workers CPU, cociente2.156x: no es aceleración CUDA. El test final continúa CPU y no mejoró. La suma comparable de preparación, entrenamiento CNN-B y test pasa825.297 a429.134s:1.923x, pero no es el pipeline completo. Como cálculo aparte, sumar los tres entrenamientos da958.806 a454.792s y2.108x, sin medir un pipeline continuo ni añadir los cocientes individuales.','paso-6-revision-02/comparacion_fases.csv y tres medias de fit');
 table(s,['Fase / alcance','Referencia (s)','Variante (s)','Cociente','Qué lo explica'],D.phasecompare.map(r=>[r.descripcion,n(r.media_cpu_s,3),n(r.media_variante_cuda_s,3),n(r.speedup,3)+'×',r.fase==='preparacion_cpu_s'?'CPU: 1 → 6 workers':r.fase==='evaluacion_cpu_s'?'Ambas en CPU':r.fase==='suma_comparable_s'?'Suma derivada; solo CNN-B':'CUDA + transferencias + val.']),.7,1.75,11.95,[3.05,1.85,1.85,1.45,3.75],16,.68);
 const sum=(key)=>['CNN-B','FC-A','FC-B'].reduce((a,m)=>a+(+row(m)[key]),0),a=sum('cpu_media_s'),b=sum('cuda_media_s');
 stat(s,n(a,2)+' s','suma de tres entrenamientos CPU',.8,5.0,5.8,'Comparación derivada del mismo alcance',C.accent2);
 stat(s,n(b,2)+' s','suma con entrenamiento CUDA',7.05,5.0,5.5,'Cociente agregado: '+n(a/b,3)+'×',C.accent3);
}
{
 const s=slide('6 · ¿Dónde se va el tiempo con CUDA?','El gráfico normaliza cada entrenamiento a100% para comparar proporciones; los tiempos absolutos de las tres redes son muy diferentes. CNN-B pasa la mayor parte del intervalo en matemática del stream. En las FC crecen las proporciones de validación CPU y administración. H2D/D2H son costes incluidos. Los eventos del stream pueden contener huecos de envío del host; no significan tiempo activo exclusivo de un kernel. La tabla muestra segundos para evitar confundir porcentaje con tiempo absoluto.','Desglose CNN-B paso6 y ampliación FC/desglose.csv');
 const cb=Object.fromEntries(D.cnn_breakdown.map(r=>[r.componente,+r.media_s]));
 const vals=[['CNN-B',cb.computo_intervalos_stream_s,cb.h2d_eventos_s,cb.d2h_eventos_s,cb.validacion_anidada_s,cb.administracion_y_staging_s],...D.breakdown.map(r=>[r.modelo,+r.matematica_intervalos_stream_s,+r.h2d_eventos_s,+r.d2h_eventos_s,+r.validacion_cpu_s,+r.resto_host_s])];
 const labels=['Matemática / stream','H2D','D2H','Validación CPU','Resto host'];
 chart(s,pres.ChartType.bar,labels.map((name,j)=>({name,labels:vals.map(r=>r[0]),values:vals.map(r=>100*r[j+1]/r.slice(1).reduce((a,b)=>a+b,0))})),.7,1.6,11.95,2.45,{barDir:'bar',barGrouping:'stacked',catAxisLabelFontSize:18,valAxisMaxVal:100,valAxisMinVal:0,valAxisLabelFormatCode:'0"%"',showLegend:true,legendPos:'b',chartColors:[THEME.colors.accent1,THEME.colors.accent5,THEME.colors.accent4,CPU,'E69F00']});
 table(s,['Modelo','Stream (s)','H2D (s)','D2H (s)','Val. CPU (s)','Host (s)'],vals.map(r=>[r[0],...r.slice(1).map(x=>n(x,3))]),.7,4.65,11.95,[1.7,2.05,1.75,1.75,2.25,2.45],16,.48);
 note(s,'Componentes anidados: NO se suman otra vez al tiempo total · Stream ≠ cómputo activo exclusivo');
}
function curve(s,model,key,x,y,w,h){const f=D.curves.filter(r=>r.modelo===model),pct=key==='val_accuracy_pct';
 chart(s,pres.ChartType.line,['CPU','CUDA'].map(name=>({name,labels:Array.from({length:15},(_,i)=>String(i+1)),values:f.filter(r=>r.origen===name).map(r=>+r[key])})),x,y,w,h,{showLegend:true,legendPos:'b',showTitle:true,title:model+' · '+(pct?'exactitud validación (%)':'pérdida validación'),titleFontFace:'+mn-lt',titleFontSize:17,catAxisTitle:'Época',showCatName:false,showCatName:false,showCatName:false,catAxisLabelFontSize:11,valAxisLabelFontSize:12,valAxisMinVal:0,...(pct?{valAxisMaxVal:100,valAxisLabelFormatCode:'0"%"'}:{valAxisLabelFormatCode:'0.0'})});}
{
 const s=slide('6 · CNN-B: aprendizaje durante 15 épocas','Los trazos son medias de cinco trayectorias por implementación y se calculan a partir de historiales guardados. AzulCPU, naranjaCUDA. La validación se ejecuta en CPU en ambas variantes, con los pesos exportados cuando el entrenamiento es CUDA. Los resultados no son una ejecución adicional. Curvas parecidas no prueban identidad de gradientes ni de máscaras; se reporta la variación de test aparte.','CNN-B: historiales.json CPU01–05 y epocas.csv GPU01–05');
 curve(s,'CNN-B','val_accuracy_pct',.6,1.65,6.05,4.55);curve(s,'CNN-B','val_loss',6.75,1.65,5.95,4.55);
 note(s,'Media de 5 corridas por implementación · pesos iniciales iguales · RNG interno distinto',.75,6.45,12,15);
}
{
 const s=slide('6 · FC-A y FC-B: curvas comparables','Arriba FC-A y abajo FC-B. Cada panel muestra cinco corridas promediadas. Dado que el estado y PCG64 se reinician, las cinco trayectorias CUDA son numéricamente repetibles: esto sirve para medir tiempo bajo condiciones similares, pero no estima variación por inicializaciones independientes. La exactitud predictiva de las FC es menor que la de CNN-B y no se mejoró artificialmente cambiando épocas o semillas después de ver resultados.','ampliación FC/epocas-comparacion.csv; curvas agregadas en datos-presentacion.json');
 curve(s,'FC-A','val_accuracy_pct',.6,1.45,6.05,2.45);curve(s,'FC-A','val_loss',6.75,1.45,5.95,2.45);
 curve(s,'FC-B','val_accuracy_pct',.6,4.0,6.05,2.45);curve(s,'FC-B','val_loss',6.75,4.0,5.95,2.45);
 note(s,'Azul: CPU histórica  ·  Naranja: CUDA nueva  ·  15 épocas fijas; no EarlyStopping',.75,6.72,12,12);
}
{
 const s=slide('6 · Más rápido no significa más exacto','Separar la métrica de tiempo de la métrica predictiva. CNN-B CUDA alcanza98.6207% en test. Su CPU histórica tuvo resultados variables y no se puede atribuir la diferencia al hardware. FC-A baja de53.065% a51.724%, yFC-B de44.291% a39.962%. Las diferencias son1.341 y4.330 puntos porcentuales. Datos, estado inicial y pasos coinciden, pero no el orden interno ni las máscaras ni todo el redondeo. La implementación demostró aceleración, no mejora predictiva.','Calidad FC y calidad_comparada.csv CNN-B; medias de5');
 const names=['CNN-B','FC-A','FC-B'];chart(s,pres.ChartType.bar,['CPU','CUDA'].map(origin=>({name:origin,labels:names,values:names.map(m=>+D.quality.find(r=>r.modelo===m&&r.origen===origin).accuracy_media_pct)})),.7,1.55,7.15,4.8,{barDir:'col',barGrouping:'clustered',valAxisMinVal:0,valAxisMaxVal:100,showValue:true,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.0"%"',showLegend:true,legendPos:'b'});
 stat(s,'98.62%','test CNN-B CUDA',8.35,1.9,4.0,'Modelo con mejor desempeño de estas redes',C.accent1);
 const qa=names.slice(1).map(m=>D.quality.filter(r=>r.modelo===m));
 text(s,'FC-A: −1.34 pp\nFC-B: −4.33 pp',8.35,4.5,4.1,1.1,25,C.accent3,true);note(s,'Diferencia vs. media CPU',8.35,5.8,4.0,14);
 note(s,'La comparación usa CPU histórica; no son trayectorias CPU/GPU pareadas');
}
{
 const s=slide('6 · Por qué funciona y qué lo limita','Responder las tres preguntas técnicas de la guía. Archivos enCPU: muchas operaciones de sistema y tareas independientes por imagen; los workers explotan esa independencia. Matemática enGPU: muchas operaciones similares y productos matriciales reutilizan paralelismo. Limitaciones: preparación y validación CPU, tamaños de lote, transferencias, sincronizaciones, memoria, coste de lanzamiento y carga histórica no pareada. No afirmar que GPU es siempre superior ni que la ocupaciónSM fue medida.','Discusión técnica y desglose medido de las tres redes');
 const a=[{title:'CPU: archivos',color:C.accent2,items:['Rutas y sistema de archivos','Decodificar y redimensionar','6 imágenes/tareas a la vez','17.32 s → 8.03 s']},
 {title:'GPU: matemática',color:C.accent3,items:['Matrices y tensores grandes','Muchos cálculos independientes','Numba-CUDA + cuBLAS','1.92× · 3.35× · 4.86×']},
 {title:'Límites reales',color:C.accent4,items:['H2D / D2H y sincronización','Validación todavía en CPU','Menos trabajo por lote en FC','RNG y referencia histórica']}];
 a.forEach((q,i)=>{const x=.7+i*4.08;box(s,x,1.75,3.7,4.35);blocks(s,x+.25,2.0,5,2,q.color,.16,.05);text(s,q.title,x+.25,2.8,3.2,.55,25,q.color,true);q.items.forEach((r,j)=>text(s,r,x+.25,3.65+j*.53,3.2,.42,16,C.text1,j===3));});
 note(s,'Tiempo menor es una medición; la causa microarquitectónica exacta requeriría un profiler',.75,6.48,12,14);
}
{
 const s=slide('De cada resultado a su evidencia','Mostrar la cadena de trazabilidad. Los mismos archivos y tensores están identificados por SHA-256; cada ejecución conserva configuración, semillas, logs, métricas y checkpoint. Los notebooks tienen salidas nativas junto al código:92celdas y30PNG CUDA; CPU37y10. Los intentos fallidos se conservan y la primera FC-A ya guardada se recuperó sin reentrenarla. El informe fue editado después por el usuario y esta presentación no lo cambia.','Manifiestos, notebooks finales, configuración y validaciones históricas; presentación usa copias de evidencia');
 const chain=[['Datos','SHA-256'],['Configuración','EXP + semillas'],['Ejecución','RUN + logs'],['Resultado','CSV + modelo'],['Evidencia','Capturas + notebook']];
 chain.forEach((q,i)=>{const x=.7+i*2.45;box(s,x,1.85,2.1,1.1);text(s,q[0],x+.15,2.02,1.8,.35,20,C.accent1,true);text(s,q[1],x+.15,2.55,1.8,.25,13,C.text2);if(i<4)line(s,x+2.13,2.4,.25,0);});
 box(s,.7,3.85,5.75,1.85);text(s,'Notebook CPU',.95,4.07,5.2,.5,27,C.accent2,true);text(s,'37 celdas de código  ·  10 imágenes',.95,4.8,5.2,.4,20,C.text1);text(s,'Conservado sin reentrenar',.95,5.35,5.2,.25,14,C.text2);
 box(s,6.85,3.85,5.75,1.85);text(s,'Notebook CUDA',7.1,4.07,5.2,.5,27,C.accent3,true);text(s,'92 celdas de código  ·  30 imágenes',7.1,4.8,5.2,.4,20,C.text1);text(s,'CNN-B + FC-A + FC-B · salidas reales',7.1,5.35,5.2,.25,14,C.text2);
 note(s,'Tiempos, gráficas y capturas reutilizados: esta presentación no ejecuta entrenamiento',.75,6.4,12,16);
}
{
 const s=slide('Tres redes completas, una conclusión','Cerrar con los tres speedups. El aporte consiste en seleccionar la matemática repetitiva, separar CPU/GPU, conservar condiciones de carga comparables y medir los costes completos. Preparación y validaciónCPU no desaparecen. No se declara aceleración del pipeline adaptativo original ni superioridad predictiva. Las conclusiones son locales a estos datos, hardware y protocolo. Abrir preguntas y usar los anexos para los detalles.','Comparación final n=5; límites documentados',true);
 ['CNN-B','FC-A','FC-B'].forEach((m,i)=>stat(s,n(row(m).speedup,2)+'×',m,.75+i*4.08,2.0,3.75,n(row(m).reduccion_pct,2)+'% menos tiempo',C.accent1,true));
 text(s,'CPU prepara  →  GPU entrena  →  CPU evalúa',.75,4.75,11.7,.7,31,C.background1,true);
 text(s,'Aceleración real · transferencias incluidas · calidad y límites visibles',.75,5.85,11.7,.6,22,C.accent6);
}
sec('Anexo para preguntas');
{
 const s=slide('A1 · Hardware, librerías y memoria','Usar como respaldo de entorno. La arena CUDA propia de las FC excluye contexto y allocations internas de bibliotecas. Para CNN-B se observa1,213MiB deVRAM y usaworkspaceim2col576MiB. La GPU es sm75; no se aplicóTF32 ni mixedprecision. Python y versiones provienen del entorno medido, no de información actual de internet.','Hardware y configuraciones congeladas; validación FC');
 table(s,['Componente','Configuración medida'],[
 ['CPU','AMD Ryzen 5 4600H · 6 físicos / 12 lógicos'],['RAM / GPU','15.42 GiB · GTX 1650 / 4 GiB / sm_75'],['CUDA / driver','Toolkit 13.1 · driver 591.59'],
 ['Python / TensorFlow','PCUDA · Python 3.14.3 · TF 2.22.0rc0 (CPU)'],['Numba / numba-cuda','0.67.0 / 0.30.4'],['Matemática GPU','cuBLAS SGEMM FP32 PEDANTIC · un stream'],
 ['Memoria FC-A / FC-B','49.46 MiB / 124.65 MiB de buffers propios'],['Convoluciones CNN-B','Workspace im2col: 576 MiB'],['Límites CPU','Intra-op 6 · inter-op 1 · BLAS 6']
 ],.7,1.55,11.95,[3.45,8.5],17,.48);
}
{
 const s=slide('A2 · Las 14 fases de referencia CPU','Son medias de cinco observaciones válidas por fase. Las cuatro fases de entrenamiento contienen su validación anidada. Para la referencia común se excluyen la rama aumentada, inferencia y las tareas que cambian según la arquitectura seleccionada. La suma es1,119.9275623s y no debe mezclarse con el pipeline adaptativo total.','EXP-002 resumen.csv y referencia común');
 const phases=['preparacion_cpu','entrenamiento_CNN-A','evaluacion_CNN-A','entrenamiento_CNN-B','evaluacion_CNN-B','entrenamiento_FC-A','evaluacion_FC-A','entrenamiento_FC-B','evaluacion_FC-B','pca','entrenamiento_RF-A','validacion_RF-A','entrenamiento_RF-B','validacion_RF-B'];
 table(s,['Fase CPU','Media (s)','DE (s)'],phases.map(p=>{const r=D.baseline.find(x=>x.fase===p);return[p.replaceAll('_',' '),n(r.media_s,3),n(r.desviacion_muestral_s,3)];}),.7,1.45,11.95,[7.0,2.5,2.45],14,.31);
 note(s,'Total común: 1,119.928 s · las validaciones anidadas NO se suman de nuevo',.75,6.75,12,12);
}
{
 const s=slide('A3 · Test final: ambas variantes usan CPU','El test mide inferencia CPU con los pesos de cada versión. Las pequeñas diferencias no representan speedup de inferenciaGPU. Todas las fases test se ejecutan con2,610imágenes ybatch32. Este cuadro separa qué sí fue acelerado de qué solamente se ejecutó después del entrenamiento CUDA.','comparacion/ampliacion-fc/comparacion.csv, fases evaluacion');
 table(s,['Test del modelo','Referencia CPU (s)','Tras fit CUDA (s)','Cociente CPU / variante'],['CNN-B','FC-A','FC-B'].map(m=>{const r=row(m,'evaluacion_');return[m,n(r.cpu_media_s,4),n(r.cuda_media_s,4),n(r.speedup,3)+'×'];}),.7,1.95,11.95,[2.8,3.0,3.05,3.1],18,.8);
 box(s,.7,5.25,11.95,.85);text(s,'No se implementó inferencia GPU en este alcance',.95,5.48,11.45,.4,25,C.accent3,true);
}
{
 const s=slide('A4 · Actividad GPU y preparación CPU','Los promedios se calculan sobre muestras etiquetadas como entrenamiento; incluyen la granularidad de muestreo de1s y no dan ocupación porSM. Las FC usan menosVRAM y menor actividad agregada, pese a mayores speedups, porque su carga y coste relativoCPU difieren. La captura pequeña documenta preparación CPU6workers en la campaña CUDAoriginal. El resto de capturas principales ya se mostró.','recursos_resumen CNN-B, recursos-resumen FC y captura preparación original');
 const a=D.cnnresources.map(r=>+r.gpu_util_media_pct),b=D.cnnresources.map(r=>+r.gpu_util_max_pct);
 const rows=[['CNN-B',n(a.reduce((x,y)=>x+y)/a.length,2)+'%',Math.max(...b)+'%','1,213'],...['FC-A','FC-B'].map(m=>{const rs=D.resources.filter(r=>r.modelo===m);return[m,n(rs.reduce((x,r)=>x+(+r.gpu_media_pct),0)/rs.length,2)+'%',Math.max(...rs.map(r=>+r.gpu_max_pct))+'%',n(Math.max(...rs.map(r=>+r.vram_max_mib)),0)];})];
 table(s,['Red','Actividad media¹','Máxima observada','VRAM máx. (MiB)'],rows,.7,1.65,11.95,[2.0,3.35,3.3,3.3],17,.58);
 picture(s,'prep',.7,3.85,6.2,2.6,'Captura preparación CPU con 6 workers');
 text(s,'Preparación CPU',7.35,4.05,5.0,.5,26,C.accent2,true);text(s,'6 workers · hasta 12 tareas pendientes',7.35,4.8,5.0,.5,18,C.text1);text(s,'17.325 s → 8.034 s  |  2.156×',7.35,5.53,5,.5,24,C.accent2,true);
 note(s,'¹ Media de porcentajes medios por RUN; muestreo 1 s · actividad agregada, no ocupación SM',.7,6.72,12,11.5);
}
{
 const s=slide('A5 · Referencias y términos de apoyo','Fuentes externas conservadas de la documentación del proyecto. Las referencias técnicas respaldan jerarquíaCUDA,cuBLASy significado de nvidia-smi; no respaldan los valores experimentales, que proceden de las corridas guardadas. Los títulos completos APA7 yURL están en estas notas. Akash.(s.f.).ASL alphabet[Conjunto de datos].Kaggle.https://www.kaggle.com/datasets/grassknoted/asl-alphabet\nNVIDIA.(s.f.-a).cuBLAS library(CUDA Toolkit13.1)[Documentación].https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html\nNVIDIA.(s.f.-b).NVIDIA System Management Interface[Documentación].https://docs.nvidia.com/deploy/nvidia-smi/index.html\nNVIDIA.(s.f.-c).WritingCUDA kernels.EnNumbaCUDA.https://nvidia.github.io/numba-cuda/user/kernels.html','Referencias primarias de informe; resultados propios EXP-002/003/011/012');
 const refs=[['Akash · ASL Alphabet','Kaggle · dataset de imágenes'],['NVIDIA · CUDA Python / Numba','Jerarquía y kernels explícitos'],['NVIDIA · cuBLAS 13.1','SGEMM y biblioteca GPU'],['NVIDIA · nvidia-smi','Monitorización de actividad y VRAM']];
 refs.forEach((q,i)=>{box(s,.7,1.65+i*1.1,6.2,.85);text(s,q[0],.92,1.8+i*1.1,5.75,.3,19,C.text1,true);text(s,q[1],.92,2.2+i*1.1,5.75,.2,12,C.text2);});
 table(s,['Término','Significado'],[['H2D / D2H','Host → Device / Device → Host'],['CNN / FC','Convolucional / totalmente conectada'],['DE / CV','Desviación estándar / variación relativa'],['Speedup','Media CPU ÷ media CUDA'],['pp','Puntos porcentuales'] ],7.35,1.65,5.25,[1.65,3.6],14,.6);
 note(s,'Datos experimentales propios · citas completas y enlaces en notas del presentador',.7,6.55,12,14);
}

async function applyTheme(filename){
 const zip=await JSZip.loadAsync(fs.readFileSync(filename));
 const escape=(s)=>s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;');
 for(const item of Object.values(zip.files).filter(f=>/^ppt\/theme\/theme\d+\.xml$/.test(f.name))){
  let xml=await item.async('string');const keys=['dk1','lt1','dk2','lt2','accent1','accent2','accent3','accent4','accent5','accent6','hlink','folHlink'];
  const scheme='<a:clrScheme name="'+escape(THEME.name)+'">'+keys.map(k=>'<a:'+k+'><a:srgbClr val="'+THEME.colors[k]+'"/></a:'+k+'>').join('')+'</a:clrScheme>';
  xml=xml.replace(/<a:clrScheme\b[^>]*>[\s\S]*?<\/a:clrScheme>/,scheme);zip.file(item.name,xml);
 }
 // pptxgenjs 4.0.1 emits a third series-axis reference for 2D bar/line
 // charts but declares that axis only for 3D. Remove this unbound reference
 // in the generator; never change measured series data or their axes.
 for(const item of Object.values(zip.files).filter(f=>/^ppt\/charts\/chart\d+\.xml$/.test(f.name))){
  let xml=await item.async('string');
  if(!xml.includes('<c:serAx>'))xml=xml.replaceAll('<c:axId val="2094734556"/>','');
  zip.file(item.name,xml);
 }
 fs.writeFileSync(filename,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
}
(async()=>{
 const target=path.join(ROOT,'entregables/Presentacion_Proyecto_3_CUDA.pptx');
 await pres.writeFile({fileName:target});await applyTheme(target);
 const dir=path.join(ROOT,'entregables/presentacion');
 fs.writeFileSync(path.join(dir,'estructura-y-notas.json'),JSON.stringify({tema:THEME,diapositivas:outline},null,2));
 fs.writeFileSync(path.join(dir,'geometria.json'),JSON.stringify({ancho:13.333333,alto:7.5,elementos:bounds},null,2));
 fs.writeFileSync(path.join(dir,'Guion_de_exposicion.md'),'# Guion de exposición\n\nLas diapositivas 1–26 forman el recorrido principal; 27–31 son respaldo para preguntas. Cada diapositiva contiene notas del presentador en PowerPoint. Se pueden abrir sin ejecutar modelos.\n\n'+outline.map(a=>'## '+a.numero+'. '+a.titulo+'\n\n'+a.notas+'\n\nProcedencia: '+a.fuentes+'\n').join('\n'));
 console.log('PRESENTACIÓN CREADA',outline.length,'diapositivas; editable; sin entrenamientos.');
})().catch(e=>{console.error(e);process.exit(1);});
