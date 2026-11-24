"""Turn a '# %%'-delimited script into a Jupyter notebook. Usage: python -I build_nb.py script.py out.ipynb"""
import json, re, sys
s = open(sys.argv[1]).read()
cells = []
for blk in re.split(r'^# %%', s, flags=re.M)[1:]:
    head, _, body = blk.partition('\n')
    if '[markdown]' in head:
        src = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.strip('\n').split('\n'))
        cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.strip('\n')})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(sys.argv[2], 'w'), indent=1)
compile('\n'.join(c['source'] for c in cells if c['cell_type'] == 'code'), 'nb', 'exec')
print('built', sys.argv[2], [c['cell_type'] for c in cells])
