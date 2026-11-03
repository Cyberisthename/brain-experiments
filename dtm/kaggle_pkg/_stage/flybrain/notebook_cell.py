# Kaggle notebook cell — attach dtm dataset (zip or extracted). GPU on.
# Handles Kaggle auto-extract: search /kaggle/input for the zip OR the run script; pick shortest path.
import os, shutil, zipfile, subprocess, sys
from pathlib import Path

IN = Path("/kaggle/input")
WORK = Path("/kaggle/working")

zips = list(IN.rglob("dtm_kaggle.zip"))
scripts = list(IN.rglob("run_dtm_kaggle.sh"))

def shortest(paths):
    return min(paths, key=lambda p: len(p.as_posix())) if paths else None

zpath = shortest(zips)
spath = shortest(scripts)
print("candidates zip=", zpath, "script=", spath)

dst = WORK / "flybrain"
if dst.exists():
    shutil.rmtree(dst)

if zpath is not None and (spath is None or len(zpath.as_posix()) <= len(spath.as_posix())):
    print("using zip", zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(WORK)
elif spath is not None:
    # already extracted tree — copy parent flybrain-like root
    # script at .../flybrain/run_dtm_kaggle.sh or .../run_dtm_kaggle.sh
    print("using extracted script", spath)
    root = spath.parent
    # if script is inside flybrain/, copy that folder as WORK/flybrain
    if root.name == "flybrain":
        shutil.copytree(root, dst)
    elif (root / "dtm").is_dir():
        shutil.copytree(root, dst)
    else:
        # script next to dtm/
        dst.mkdir(parents=True)
        shutil.copytree(root, dst / root.name) if False else None
        # fallback: copy whole dataset dir
        shutil.copytree(spath.parents[0], dst, dirs_exist_ok=True)
else:
    raise SystemExit("dtm_kaggle.zip or run_dtm_kaggle.sh not found under /kaggle/input")

script = dst / "run_dtm_kaggle.sh"
if not script.exists():
    # maybe nested
    found = list(dst.rglob("run_dtm_kaggle.sh"))
    assert found, sorted(p.name for p in dst.rglob("*"))[:50]
    # use shortest
    script = min(found, key=lambda p: len(p.as_posix()))
    # ROOT for bash is script.parent
print("running", script)

env = os.environ.copy()
env["OUT_DIR"] = str(WORK / "out")
proc = subprocess.run(["bash", str(script)], cwd=str(script.parent), env=env)
print("exit", proc.returncode)
sys.exit(proc.returncode)
