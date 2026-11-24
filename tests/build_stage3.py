"""Insert the Stage 2 function cell into the Stage 3 template. Usage: python -I build_stage3.py <pkg_dir>"""
import sys

pkg = sys.argv[1]
s2 = open(f"{pkg}/stage2_train_pcam.py").read()
a = s2.index("# %%\nimport gc")
b = s2.index("# %%\ndef main():")
shared = s2[a + len("# %%\n"):b].rstrip() + "\n"
tpl = open(f"{pkg}/stage3_template.py").read()
assert tpl.count("# <<SHARED_STAGE2_CODE>>\n") == 1
out = tpl.replace("# <<SHARED_STAGE2_CODE>>\n",
                  "# Shared with Stage 2 (data loading, batching, model, preparation); copied verbatim.\n" + shared)
open(f"{pkg}/stage3_fidelity_pcam.py", "w").write(out)
print("wrote stage3_fidelity_pcam.py,", len(out.splitlines()), "lines")
