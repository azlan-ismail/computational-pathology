"""Insert the Stage 2 function cell into a template. Usage: python -I build_from_template.py <pkg> <template> <out.py>"""
import sys
pkg, tpl_name, out_name = sys.argv[1:4]
s2 = open(f"{pkg}/stage2_train_pcam.py").read()
a = s2.index("# %%\nimport gc"); b = s2.index("# %%\ndef main():")
shared = s2[a + len("# %%\n"):b].rstrip() + "\n"
tpl = open(f"{pkg}/{tpl_name}").read()
assert tpl.count("# <<SHARED_STAGE2_CODE>>\n") == 1
open(f"{pkg}/{out_name}", "w").write(tpl.replace("# <<SHARED_STAGE2_CODE>>\n",
    "# Shared with Stage 2 (data loading, batching, model, preparation); copied verbatim.\n" + shared))
print("wrote", out_name)
