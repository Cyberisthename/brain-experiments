# Kaggle notebook cell — dataset slug: lanneryen/engine-v3-ablation
# Accelerator: GPU. No pip install required (torch + scipy are preinstalled).
# Works whether Kaggle leaves the zip as-is OR auto-extracts the dataset tree.
import os, shutil, zipfile, subprocess, sys
from pathlib import Path

IN_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def _shortest(paths):
    paths = [Path(p) for p in paths]
    if not paths:
        return None
    return min(paths, key=lambda p: (len(p.parts), len(str(p))))


# Prefer already-extracted run script; else the zip. Shortest path wins ties.
script_cands = list(IN_ROOT.rglob("run_ablation_kaggle.sh"))
zip_cands = list(IN_ROOT.rglob("engine_v3_kaggle.zip"))
script = _shortest(script_cands)
zpath = _shortest(zip_cands)
print("input script_cands", len(script_cands), "zip_cands", len(zip_cands))
print("chosen script", script)
print("chosen zip", zpath)

dst = WORK / "flybrain"
if dst.exists():
    shutil.rmtree(dst)

if script is not None and script.name == "run_ablation_kaggle.sh":
    # Dataset already expanded (zip contents or flat tree). Copy flybrain root.
    # run script may live at .../flybrain/run_ablation_kaggle.sh or dataset root.
    root = script.parent
    if (root / "engine_v3").is_dir():
        src_root = root
    elif (root / "flybrain" / "engine_v3").is_dir():
        src_root = root / "flybrain"
    else:
        # walk up a level if script is nested oddly
        src_root = root
    print("using extracted tree", src_root)
    shutil.copytree(src_root, dst, symlinks=False)
elif zpath is not None:
    print("extracting zip", zpath)
    with zipfile.ZipFile(zpath, "r") as zf:
        zf.extractall(WORK)
    # zip top-level is flybrain/
    if not (WORK / "flybrain" / "run_ablation_kaggle.sh").exists():
        # sometimes zip is nested under an extra folder
        found = _shortest((WORK).rglob("run_ablation_kaggle.sh"))
        assert found, sorted(p.name for p in WORK.iterdir())
        if found.parent != dst:
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(found.parent), str(dst))
else:
    raise SystemExit(
        f"Neither engine_v3_kaggle.zip nor run_ablation_kaggle.sh under {IN_ROOT}"
    )

assert (dst / "run_ablation_kaggle.sh").exists(), sorted(p.name for p in dst.iterdir())
print("flybrain ready at", dst)

env = os.environ.copy()
env["OUT_DIR"] = str(WORK / "out")
proc = subprocess.run(
    ["bash", str(dst / "run_ablation_kaggle.sh")],
    cwd=str(dst),
    env=env,
)
print("exit", proc.returncode)
sys.exit(proc.returncode)
