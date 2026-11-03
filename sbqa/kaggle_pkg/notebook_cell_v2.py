import os, shutil, zipfile, subprocess, sys
from pathlib import Path
IN = Path("/kaggle/input"); WORK = Path("/kaggle/working"); dst = WORK/"flybrain"
if dst.exists(): shutil.rmtree(dst)
z = list(IN.rglob("sbqa_kaggle.zip"))
if z:
    zipfile.ZipFile(z[0]).extractall(WORK)
else:
    sh = sorted(IN.rglob("run_sbqa_kaggle.sh"), key=lambda p: len(str(p))); assert sh, list(IN.rglob("*"))[:50]
    shutil.copytree(sh[0].parent, dst)
assert (dst/"run_sbqa_kaggle.sh").exists(), sorted(p.name for p in WORK.rglob("*"))[:50]
env = os.environ.copy(); env["OUT_DIR"] = str(WORK/"out")
p = subprocess.run(["bash", str(dst/"run_sbqa_kaggle.sh")], cwd=str(dst), env=env)
print("exit", p.returncode)
