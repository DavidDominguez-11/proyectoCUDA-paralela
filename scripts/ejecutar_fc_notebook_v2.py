"""Execute only EXP-011 FC cells, never any existing experiment/notebook."""
from pathlib import Path
import os,sys,json,traceback
from datetime import datetime,timezone
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'resultados/paralelo/EXP-011'
def main():
    output=BASE/'FC_CUDA.ejecutado.ipynb'
    if output.exists():raise FileExistsError(output)
    nb=nbformat.read(BASE/'FC_CUDA.plantilla.ipynb',as_version=4)
    specs=ROOT/'data/jupyter/kernels'
    os.environ['IPYTHONDIR']=str(ROOT/'data/jupyter/ipython')
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'data/jupyter/runtime')
    manager=KernelSpecManager(kernel_dirs=[str(specs)],ensure_native_kernel=False)
    km=KernelManager(kernel_name='pcuda',kernel_spec_manager=manager)
    state=dict(inicio_utc=datetime.now(timezone.utc).isoformat(),estado='ejecutando',celdas_terminadas=0,
        comando=[sys.executable,*sys.argv],metodo='NotebookClient; código FC real visible y ejecutado por celdas')
    def save():
        nbformat.write(nb,output)
        (BASE/'estado-notebook.json').write_text(json.dumps(state,indent=2),encoding='utf8')
    def done(cell,cell_index,**kw):
        state.update(celdas_terminadas=cell_index+1,ultima_celda=cell.source[:110],actualizado_utc=datetime.now(timezone.utc).isoformat());save()
    try:
        save()
        NotebookClient(nb,km=km,timeout=10800,resources={'metadata':{'path':str(ROOT)}},on_cell_executed=done).execute(cwd=str(ROOT))
        state['estado']='completada'
    except BaseException:
        state.update(estado='fallida',error=traceback.format_exc());raise
    finally:
        state['fin_utc']=datetime.now(timezone.utc).isoformat();save()
        if km.has_kernel:km.shutdown_kernel(now=True)
        for stream in ['stdout','stderr']:
            with (BASE/(stream+'.log')).open('x',encoding='utf8') as f:
                for cell in nb.cells:
                    for out in cell.get('outputs',[]):
                        if out.output_type=='stream' and out.name==stream:f.write(out.text)
    print('FIN',state['estado'],flush=True)
if __name__=='__main__':main()
