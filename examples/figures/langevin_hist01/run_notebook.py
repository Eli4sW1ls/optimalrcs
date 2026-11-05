"""Execute a notebook in place with nbclient (nbconvert is broken in this env), saving the outputs."""
import sys, nbformat
from nbclient import NotebookClient
path = sys.argv[1]
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": sys.argv[2]}}).execute()
nbformat.write(nb, path)
print("NOTEBOOK DONE", flush=True)
