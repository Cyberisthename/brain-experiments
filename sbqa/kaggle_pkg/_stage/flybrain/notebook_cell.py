# Kaggle notebook cell — dataset slug: lanneryen/sbqa-flybrain (or whatever you name it)
# Accelerator: GPU. No pip install required (torch + scipy are preinstalled).
import os, shutil, zipfile, subprocess, sys
from pathlib import Path

IN = Path("/kaggle/input")
# find zip under any attached dataset
cands = list(IN.rglob("sbqa_kaggle.zip"))
assert cands, f"sbqa_kaggle.zip not found under {IN}"
zpath = cands[0]
print("using zip", zpath)

WORK = Path("/kaggle/working")
dst = WORK / "flybrain"
if dst.exists():
    shutil.rmtree(dst)
with zipfile.ZipFile(zpath, "r") as zf:
    zf.extractall(WORK)
assert (WORK / "flybrain" / "run_sbqa_kaggle.sh").exists(), sorted(p.name for p in WORK.iterdir())

env = os.environ.copy()
env["OUT_DIR"] = str(WORK / "out")
proc = subprocess.run(
    ["bash", str(WORK / "flybrain" / "run_sbqa_kaggle.sh")],
    cwd=str(WORK / "flybrain"),
    env=env,
)
print("exit", proc.returncode)
sys.exit(proc.returncode)
