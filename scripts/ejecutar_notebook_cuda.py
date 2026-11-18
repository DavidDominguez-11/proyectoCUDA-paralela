"""Execute the notebook with a workspace-local PCUDA kernelspec; preserve output."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--entrada",type=Path,default=ROOT/"notebooks/Lab3_ASL_CUDA.ipynb")
    ap.add_argument("--salida",required=True,type=Path)
    args=ap.parse_args()
    if args.salida.exists():
        raise FileExistsError("An executed notebook is immutable; choose another output")
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    specs=ROOT/"data/jupyter/kernels"
    (specs/"pcuda").mkdir(parents=True,exist_ok=True)
    (specs/"pcuda/kernel.json").write_text(json.dumps(dict(
        argv=[sys.executable,"-m","ipykernel_launcher","--HistoryManager.hist_file=:memory:","-f","{connection_file}"],
        display_name="PCUDA",language="python")),encoding="utf8")
    os.environ["ASL_CUDA_RUN"]="1"
    os.environ["JUPYTER_RUNTIME_DIR"]=str(ROOT/"data/jupyter/runtime")
    os.environ["IPYTHONDIR"]=str(ROOT/"data/jupyter/ipython")
    os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"]="0"
    manager=KernelSpecManager(kernel_dirs=[str(specs)],ensure_native_kernel=False)
    km=KernelManager(kernel_name="pcuda",kernel_spec_manager=manager)
    notebook=nbformat.read(args.entrada,as_version=4)
    client=NotebookClient(notebook,km=km,timeout=14400,resources={"metadata":{"path":str(ROOT)}})
    try:
        client.execute(cwd=str(ROOT))
    finally:
        args.salida.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(notebook,args.salida)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    print("Notebook ejecutado:",args.salida,flush=True)


if __name__=="__main__":
    main()
