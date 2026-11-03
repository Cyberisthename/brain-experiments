"""Training-free model soup: uniform average of compatible checkpoints.

Only full medium-arch state dicts with identical keys/shapes are averaged.
If a single compatible ckpt is found, the soup is identical to that ckpt
(honest N=1; still exercises the load path for eval).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

from model_io import DEFAULT_CKPT, DEFAULT_DATA, load_medium


def _state_dict(blob) -> dict:
    if isinstance(blob, dict) and "model" in blob:
        return blob["model"]
    if isinstance(blob, dict) and any(k.startswith("tok_emb") or k.startswith("blocks.") for k in blob):
        return blob
    raise ValueError("checkpoint has no model state dict")


def discover_medium_ckpts(
    roots: Optional[Sequence[Path]] = None,
    ref: Path = DEFAULT_CKPT,
) -> List[Path]:
    """Walk roots for .pt files whose tok_emb shape + key set match `ref`."""
    roots = list(roots or [
        Path("/workspace/jarvis/checkpoints"),
        Path("/workspace/jarvis_kaggle"),
    ])
    ref_blob = torch.load(ref, map_location="cpu", weights_only=False)
    ref_sd = _state_dict(ref_blob)
    ref_keys = set(ref_sd.keys())
    ref_shape = tuple(ref_sd["tok_emb.weight"].shape)
    found: List[Path] = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.pt"):
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            try:
                blob = torch.load(p, map_location="cpu", weights_only=False)
                sd = _state_dict(blob)
            except Exception:
                continue
            if "tok_emb.weight" not in sd:
                continue
            if tuple(sd["tok_emb.weight"].shape) != ref_shape:
                continue
            if set(sd.keys()) != ref_keys:
                continue
            found.append(p.resolve())
    # stable order; prefer medium_v3/best first
    found = sorted(found, key=lambda p: (0 if "medium_v3" in str(p) and "best" in p.name else 1, str(p)))
    return found


def average_state_dicts(paths: Sequence[Path]) -> Tuple[dict, dict]:
    """Uniform average of floating tensors; non-floats taken from first ckpt.

    Returns (averaged_sd, meta).
    """
    if not paths:
        raise ValueError("no checkpoints to average")
    blobs = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    sds = [_state_dict(b) for b in blobs]
    keys = set(sds[0].keys())
    for sd in sds[1:]:
        keys &= set(sd.keys())
    out = {}
    for k in sorted(keys):
        vals = [sd[k] for sd in sds]
        if all(torch.is_tensor(v) and v.is_floating_point() for v in vals):
            acc = vals[0].float().clone()
            for v in vals[1:]:
                acc += v.float()
            out[k] = (acc / float(len(vals))).to(dtype=vals[0].dtype)
        else:
            out[k] = vals[0]
    steps = [int(b.get("step", -1)) if isinstance(b, dict) else -1 for b in blobs]
    meta = {
        "n": len(paths),
        "paths": [str(p) for p in paths],
        "steps": steps,
        "identical_to_single": len(paths) == 1,
        "note": (
            "N=1 soup ≡ that checkpoint (only one medium-compatible ckpt on disk)"
            if len(paths) == 1
            else f"uniform average of {len(paths)} compatible medium checkpoints"
        ),
    }
    return out, meta


def save_soup(out_path: Path, paths: Optional[Sequence[Path]] = None) -> dict:
    paths = list(paths) if paths is not None else discover_medium_ckpts()
    if not paths:
        paths = [DEFAULT_CKPT]
    sd, meta = average_state_dicts(paths)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "model": sd,
        "step": int(sum(meta["steps"]) / max(1, len(meta["steps"]))),
        "meta": {
            "soup": meta,
            "arch": "medium",
            "model_only": True,
            "fp16": False,
        },
    }
    torch.save(blob, out_path)
    meta["out"] = str(out_path)
    return meta


def load_soup_medium(
    soup_path: Optional[Path] = None,
    max_ctx: Optional[int] = None,
    rebuild: bool = True,
):
    """Build (or reuse) soup ckpt under chat_rag/results/medium_soup.pt and load it."""
    here = Path(__file__).resolve().parent
    soup_path = Path(soup_path) if soup_path else here / "results" / "medium_soup.pt"
    meta = None
    if rebuild or not soup_path.exists():
        meta = save_soup(soup_path)
    model, tok, info = load_medium(ckpt=soup_path, data_dir=DEFAULT_DATA, max_ctx=max_ctx)
    info["soup"] = meta or {"out": str(soup_path), "note": "reused existing soup file"}
    return model, tok, info


if __name__ == "__main__":
    meta = save_soup(Path(__file__).resolve().parent / "results" / "medium_soup.pt")
    print(meta)
