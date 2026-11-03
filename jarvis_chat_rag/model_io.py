"""Load medium_v3 and generate (CPU, ≤2 threads). Does not modify kaggle/chatbot trees."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import torch
import torch.nn.functional as F

torch.set_num_threads(2)

KAGGLE = Path("/workspace/jarvis_kaggle")
sys.path.insert(0, str(KAGGLE))

from model.bpe_tokenizer import BPETokenizer
from model.transformer import JarvisTransformer, PRESETS, count_parameters

DEFAULT_CKPT = Path("/workspace/jarvis/checkpoints/medium_v3/best.pt")
DEFAULT_DATA = KAGGLE / "bundle_medium"


def load_medium(
    ckpt: Path = DEFAULT_CKPT,
    data_dir: Path = DEFAULT_DATA,
    max_ctx: Optional[int] = None,
) -> Tuple[JarvisTransformer, BPETokenizer, dict]:
    tok = BPETokenizer.load(data_dir / "tokenizer.json")
    arch = dict(PRESETS["medium"])
    model = JarvisTransformer(vocab_size=tok.vocab_size, attn="plain", dropout=0.0, **arch)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = dict(blob.get("meta") or {})
    sd = blob["model"] if "model" in blob else blob
    sd = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    if max_ctx:
        model.max_seq_len = min(model.max_seq_len, int(max_ctx))
    info = {
        "ckpt": str(ckpt),
        "step": int(blob.get("step", meta.get("step", -1))),
        "n_params": count_parameters(model, trainable_only=False),
        "max_seq_len": model.max_seq_len,
        "missing": len(missing),
        "unexpected": len(unexpected),
        "meta_eval_loss": meta.get("eval_loss"),
    }
    return model, tok, info


@torch.no_grad()
def generate(
    model: JarvisTransformer,
    tok: BPETokenizer,
    prompt: str,
    max_new: int = 48,
    temperature: float = 0.0,
    top_k: int = 40,
) -> str:
    """temperature<=0 → greedy."""
    ids = tok.encode(prompt, add_eos=False)
    if len(ids) > model.max_seq_len - 4:
        ids = ids[-(model.max_seq_len - 4) :]
    x = torch.tensor([ids], dtype=torch.long)
    greedy = temperature is None or temperature <= 0
    for _ in range(max_new):
        ctx = x[:, -model.max_seq_len :]
        logits, _ = model(ctx)
        logits = logits[:, -1, :]
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / max(float(temperature), 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, nxt], dim=1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x[0].tolist())


def continuation_only(full: str, prompt: str) -> str:
    if full.startswith(prompt):
        return full[len(prompt) :]
    # fallback: after last Answer: / Assistant:
    for marker in ("\nAnswer:", "\nAssistant:", "Answer:", "Assistant:"):
        if marker in full:
            return full.split(marker)[-1]
    return full
