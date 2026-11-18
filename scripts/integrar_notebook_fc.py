"""Append actual new native FC outputs, preserving every existing CUDA cell."""
from pathlib import Path
import sys,json,hashlib
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/paralelo/EXP-012'
OUT=ROOT/'resultados/preparacion/ampliacion-fc'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    state=json.loads((BASE/'estado-notebook.json').read_text(encoding='utf8'));assert state['estado']=='completada'
    new=nbf.read(BASE/'FC_CUDA.ejecutado.ipynb',4)
    assert not [o for c in new.cells for o in c.get('outputs',[]) if o.output_type=='error']
    firstnb=nbf.read(ROOT/'resultados/paralelo/EXP-011/FC_CUDA.ejecutado.ipynb',4)
    firstcells=[]
    for cell in firstnb.cells:
        if any(o.output_type=='error' for o in cell.get('outputs',[])):break
        firstcells.append(cell)
    # The first run already completed its measured work and saved its model.
    # Execute presentation from its real files; do not train or evaluate again.
    recovery=nbf.v4.new_notebook()
    recovery.metadata=dict(kernelspec=dict(name='pcuda',display_name='PCUDA',language='python'))
    recovery.cells=[nbf.v4.new_markdown_cell('## Cierre recuperado de FC-A, corrida1 — solo lectura\n\nEl entrenamiento, test y checkpoint terminaron antes de un fallo de presentación por Styler. El notebook fallido queda archivado. Estas celdas nuevas leen las salidas guardadas sin repetir cálculos del modelo.'),nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd,json,sys\nimport matplotlib.pyplot as plt\nfrom IPython.display import display,Image\nROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))\nfrom registro_notebook import mostrar_figura\nrun=next((ROOT/'resultados/paralelo/EXP-011').glob('RUN-*-fc-a-01'))\ndest=ROOT/'resultados/preparacion/ampliacion-fc/recuperacion-fc-a-01'\ndest.mkdir(exist_ok=False)\nprint('Recuperación SIN reentrenar:',run.name)\nprint((run/'metricas.json').read_text(encoding='utf8'))\nprint((run/'modelo-guardado.json').read_text(encoding='utf8'))\ndisplay(pd.read_csv(run/'tiempos.csv'))\ne=pd.read_csv(run/'epocas.csv');display(e)"),nbf.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')\nfor ax,keys,title in zip(axes,[('train_loss','val_loss'),('train_accuracy','val_accuracy')],['Pérdida','Exactitud']):\n    for key in keys:ax.plot(e.epoca,e[key],label=key)\n    ax.set(xlabel='Época',ylabel=title,title='FC-A · corrida1');ax.legend();ax.grid(alpha=.2)\nfig.savefig(dest/'curvas.png',dpi=160);fig.savefig(dest/'curvas.svg');mostrar_figura(fig)\nr=pd.read_csv(run/'recursos.csv')\nfig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')\naxes[0].plot(r.transcurrido_s,r.cpu_procesos_pct,label='CPU proceso e hijos');axes[0].set(ylabel='CPU (%)')\naxes[1].plot(r.transcurrido_s,r.gpu_util_pct,color='#D55E00',label='GPU');axes[1].set(xlabel='Tiempo desde inicio del RUN (s)',ylabel='Actividad GPU (%)',ylim=(0,105))\nfor ax in axes:ax.legend();ax.grid(alpha=.2)\nfig.savefig(dest/'recursos.png',dpi=160);fig.savefig(dest/'recursos.svg');mostrar_figura(fig)\nfor image in run.glob('captura-*.png'):display(Image(filename=str(image)))")]
    manager=KernelSpecManager(kernel_dirs=[str(ROOT/'data/jupyter/kernels')],ensure_native_kernel=False)
    km=KernelManager(kernel_name='pcuda',kernel_spec_manager=manager)
    try:NotebookClient(recovery,km=km,timeout=120).execute(cwd=str(ROOT))
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    nbf.write(recovery,OUT/'recuperacion-fc-a-01.ejecutado.ipynb')
    # This additional kernel ONLY reads CSV/PNG files; it never imports the model.
    summary=nbf.v4.new_notebook()
    summary.metadata=dict(kernelspec=dict(name='pcuda',display_name='PCUDA',language='python'))
    summary.cells=[nbf.v4.new_markdown_cell('# Comparación ampliada: resultados CPU conservados y CUDA nueva\n\nSesión separada de presentación: solo lectura de mediciones y gráficas reales. Los contadores reinician en cada sesión; no se reejecuta ningún entrenamiento.'),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd,json\nfrom IPython.display import display,Image\nROOT=Path.cwd()\nOUT=ROOT/'resultados/comparacion/ampliacion-fc'\ncomparison=pd.read_csv(OUT/'comparacion.csv')\ndisplay(comparison)\nprint('Speedup = media CPU / media CUDA; n=5 por modelo y dispositivo. Evaluación final sigue en CPU.')"),
        nbf.v4.new_code_cell("display(pd.read_csv(OUT/'calidad.csv'))\ndisplay(pd.read_csv(OUT/'desglose.csv'))\ndisplay(pd.read_csv(OUT/'recursos-resumen.csv'))"),
        nbf.v4.new_code_cell("for name in ['01_fc_tiempos','02_tres_redes_speedup','03_fc_desglose','04_fc_calidad']:\n    print(name)\n    display(Image(filename=str(OUT/(name+'.png'))))\nprint('Datos idénticos por SHA-256; CPU histórica con distinto RNG interno. GPU util no equivale a ocupación por SM. Sin saturación artificial de VRAM.')")]
    manager=KernelSpecManager(kernel_dirs=[str(ROOT/'data/jupyter/kernels')],ensure_native_kernel=False)
    km=KernelManager(kernel_name='pcuda',kernel_spec_manager=manager)
    try:NotebookClient(summary,km=km,timeout=120).execute(cwd=str(ROOT))
    finally:
        if km.has_kernel:km.shutdown_kernel(now=True)
    nbf.write(summary,OUT/'comparacion-notebook.ejecutado.ipynb')
    backup=OUT/'ASL_CUDA.ipynb';target=ROOT/'entregables/notebooks/ASL_CUDA.ipynb'
    original=nbf.read(backup,4)
    assert sha(target)==sha(backup),'CUDA deliverable changed since backup; preserve user edits before merging'
    oldcells=json.loads(json.dumps(original.cells))
    original.cells.append(nbf.v4.new_markdown_cell('# Sesiones nuevas: extensión a FC-A y FC-B\n\nLas celdas anteriores y sus salidas se conservan íntegramente como evidencia CNN-B. Se anexan las celdas reales de FC-A1 ya guardada, su cierre recuperado de archivos y las nueve corridas restantes. Los contadores nativos reinician en cada sesión. No se volvió a ejecutar CNN-B ni el notebook CPU. EXP-010 quedó incompleto por exportaciónCSV y no integra los promedios. EXP-011 aporta FC-A1 guardada; EXP-012 ejecuta exclusivamente FC-A2–5 yFC-B1–5.'))
    original.cells.extend(firstcells)
    original.cells.extend(recovery.cells)
    original.cells.append(nbf.v4.new_markdown_cell('# Sesión nueva: las nueve corridas FC restantes\n\nLos contadores reinician en1. El bootstrap verifica los datos y fuentes, y recupera la primera métrica guardada. Solo se entrenan las nueve repeticiones pendientes.'))
    original.cells.extend(new.cells)
    # C-level kernel diagnostics bypass IOPub; preserve them literally, not as fake outputs.
    diagnostics='EXP-011 (primera corrida guardada; incluye fallo de presentación archivado):\n'+(ROOT/'resultados/paralelo/EXP-011/controlador-stderr.log').read_text(encoding='utf8')+'\n\nEXP-012 (nueve pendientes):\n'+(BASE/'controlador-stderr.log').read_text(encoding='utf8')
    original.cells.append(nbf.v4.new_markdown_cell('## Diagnóstico nativo del kernel FC\n\nSalida real del descriptor stderr del controlador/kernel; no es una celda de entrenamiento adicional.\n\n```text\n'+diagnostics+'\n```'))
    original.cells.extend(summary.cells)
    original.metadata['ampliacion_fc']=dict(experimento='EXP-012',notebook_previo_sha256=sha(backup),
        notebook_fc_nativo_sha256=sha(BASE/'FC_CUDA.ejecutado.ipynb'),
        sesiones='CNN-B previa; FC nueva; presentación solo lectura. Contadores propios por sesión.')
    nbf.write(original,target)
    assert json.loads(json.dumps(nbf.read(target,4).cells[:len(oldcells)]))==oldcells
    # CPU final must remain byte-for-byte unchanged.
    before=json.loads((OUT/'identidades-previas.json').read_text(encoding='utf8'))
    assert sha(ROOT/'entregables/notebooks/ASL_Secuencial.ipynb')==before['entregables/notebooks/ASL_Secuencial.ipynb']['sha256']
    report=dict(notebook_cuda_sha256=sha(target),celdas_previas_preservadas=len(oldcells),
        celdas_fc_nativas=len(new.cells)+len(firstcells),codigo_fc=sum(c.cell_type=='code' for c in new.cells+firstcells),
        codigo_total=sum(c.cell_type=='code' for c in original.cells),
        png_total=sum('image/png' in o.get('data',{}) for c in original.cells for o in c.get('outputs',[])),
        errores=sum(o.output_type=='error' for c in original.cells for o in c.get('outputs',[])),
        notebook_cpu_inalterado=True)
    (OUT/'integracion-notebook.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(report)
if __name__=='__main__':main()
