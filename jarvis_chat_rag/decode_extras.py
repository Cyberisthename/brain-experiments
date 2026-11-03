"""Training-free decoding helpers for RAG: CAD + retrieved-token copy bias.

CAD (Shi et al. 2023/2024): amplify (logits_with_context - logits_without_context).
Copy-bias: add a positive logit offset to any vocab id that appears in the retrieved
context text (cheap pointer-style prior; no training).
"""
from __future__ import annotations

from typing import Optional, Set

import torch
import torch.nn.functional as F


def _encode_trim(tok, text: str, max_seq_len: int) -> torch.Tensor:
    ids = tok.encode(text, add_eos=False)
    if len(ids) > max_seq_len - 4:
        ids = ids[-(max_seq_len - 4) :]
    return torch.tensor([ids], dtype=torch.long)


def _pick(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """logits: [1, V] -> next id [1, 1]."""
    if temperature is None or temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / max(float(temperature), 1e-6)
    if top_k:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate_cad(
    model,
    tok,
    prompt_with_ctx: str,
    prompt_no_ctx: str,
    max_new: int = 48,
    temperature: float = 0.0,
    top_k: int = 40,
    alpha: float = 1.0,
) -> str:
    """Context-aware decoding (CAD).

    logits' = (1+α)·logits_c − α·logits  (Shi et al.)
    Default α=1.0 (strong context trust; good for knowledge-conflict / novel facts).
    """
    x_c = _encode_trim(tok, prompt_with_ctx, model.max_seq_len)
    x = _encode_trim(tok, prompt_no_ctx, model.max_seq_len)
    for _ in range(max_new):
        ctx_c = x_c[:, -model.max_seq_len :]
        ctx = x[:, -model.max_seq_len :]
        logits_c, _ = model(ctx_c)
        logits, _ = model(ctx)
        logits_c = logits_c[:, -1, :]
        logits = logits[:, -1, :]
        cad = (1.0 + float(alpha)) * logits_c - float(alpha) * logits
        nxt = _pick(cad, temperature, top_k)
        x_c = torch.cat([x_c, nxt], dim=1)
        x = torch.cat([x, nxt], dim=1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x_c[0].tolist())


def context_token_ids(tok, context: str) -> Set[int]:
    """Unique BPE ids that appear in the retrieved context string."""
    if not context:
        return set()
    ids = tok.encode(context, add_eos=False)
    return set(int(i) for i in ids)


@torch.no_grad()
def generate_copy_bias(
    model,
    tok,
    prompt: str,
    context: str,
    max_new: int = 48,
    temperature: float = 0.0,
    top_k: int = 40,
    bias: float = 3.0,
) -> str:
    """Add `bias` to logits of any token that appears in retrieved context."""
    boost: Set[int] = context_token_ids(tok, context)
    x = _encode_trim(tok, prompt, model.max_seq_len)
    boost_idx = None
    if boost:
        boost_idx = torch.tensor(sorted(boost), dtype=torch.long)
    for _ in range(max_new):
        ctx = x[:, -model.max_seq_len :]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].clone()
        if boost_idx is not None and boost_idx.numel() > 0:
            logits[0, boost_idx] = logits[0, boost_idx] + float(bias)
        nxt = _pick(logits, temperature, top_k)
        x = torch.cat([x, nxt], dim=1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x[0].tolist())


def make_copy_prompt(question: str) -> str:
    """Few-shot-ish template that tells a tiny LM to copy from Context."""
    q = question.strip()
    return (
        "Use ONLY the Context above. Copy the shortest exact answer span.\n"
        f"Question: {q}\n"
        "Answer:"
    )


@torch.no_grad()
def generate_soft_exclusion(
    model,
    tok,
    prompt: str,
    context: str,
    max_new: int = 48,
    temperature: float = 0.8,
    top_k: int = 40,
    strength: float = 0.7,
    repeat_window: int = 24,
    unsupported_strength: Optional[float] = None,
) -> str:
    """Negative-constraint decoding (user idea): soft-attenuate excluded tokens, renormalize.

    Excluded at each step:
      1) tokens that appear in the recent `repeat_window` of generated ids (anti-loop)
      2) tokens NOT present in the retrieved context string (unsupported by passages)

    Soft rule: p'(t) ∝ p(t) * (1 - strength)  for excluded t, then renormalize.
    Not a hard ban — low-prob / excluded mass is attenuated, then the distribution
    is renormalized (mirrors soft attenuation of excluded states).

    `unsupported_strength` defaults to `strength`. Set 0 to only soft-penalize repeats.
    Compare vs plain sampling, hard top-k, and retrieved-token copy bias.
    """
    if unsupported_strength is None:
        unsupported_strength = strength
    ctx_ids = context_token_ids(tok, context)
    x = _encode_trim(tok, prompt, model.max_seq_len)
    gen_start = x.size(1)
    # keep a little mass even at strength=1
    floor = 1e-8
    for _ in range(max_new):
        ctx = x[:, -model.max_seq_len :]
        logits, _ = model(ctx)
        logits = logits[:, -1, :].clone()
        # Soft-attenuate FIRST (full vocab), THEN optional top-k on the
        # reweighted distribution — otherwise rare context tokens never enter top-k.
        if temperature is None or temperature <= 0:
            temp = 1e-6
        else:
            temp = float(temperature)
        logits = logits / max(temp, 1e-6)
        probs = F.softmax(logits, dim=-1).clamp_min(floor)

        excl = torch.zeros_like(probs, dtype=torch.bool)
        recent = x[0, max(gen_start, x.size(1) - repeat_window) :].tolist()
        if recent:
            excl[0, torch.tensor(recent, dtype=torch.long)] = True
        if ctx_ids:
            unsupported = torch.ones(probs.size(-1), dtype=torch.bool)
            supported = torch.tensor(sorted(ctx_ids), dtype=torch.long)
            unsupported[supported] = False
        else:
            unsupported = None

        mult = torch.ones_like(probs)
        if excl.any():
            mult = torch.where(excl, mult * (1.0 - float(strength)), mult)
        if unsupported is not None and float(unsupported_strength) > 0:
            mult = mult * torch.where(
                unsupported.unsqueeze(0),
                torch.full_like(mult, 1.0 - float(unsupported_strength)),
                torch.ones_like(mult),
            )
        probs = probs * mult
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(floor)

        # optional hard shortlist after soft reweight
        if top_k and top_k > 0:
            v, idx = torch.topk(probs, min(top_k, probs.size(-1)))
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            probs = probs.masked_fill(~mask, 0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(floor)

        if temperature is None or temperature <= 0:
            nxt = probs.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, nxt], dim=1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x[0].tolist())


@torch.no_grad()
def generate_plain(
    model,
    tok,
    prompt: str,
    max_new: int = 48,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
) -> str:
    """Plain sampling (top_k=None → full softmax) or hard top-k if top_k set."""
    x = _encode_trim(tok, prompt, model.max_seq_len)
    for _ in range(max_new):
        ctx = x[:, -model.max_seq_len :]
        logits, _ = model(ctx)
        logits = logits[:, -1, :]
        nxt = _pick(logits, temperature if temperature and temperature > 0 else 0.0, top_k or 0)
        # _pick with top_k=0 skips masking — good for plain
        x = torch.cat([x, nxt], dim=1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x[0].tolist())


# ---------------------------------------------------------------------------
# Passage-scoped kNN-LM (Khandelwal et al. 2020), training-free / no FAISS.
# Datastore = (hidden_t, next_token) pairs from the retrieved context only.
# ---------------------------------------------------------------------------


@torch.no_grad()
def forward_logits_and_hidden(model, input_ids: torch.Tensor):
    """Run model forward; capture ln_f hidden states via hook (no model edits)."""
    captured = {}

    def _hook(_module, _inp, out):
        captured["h"] = out

    handle = model.ln_f.register_forward_hook(_hook)
    try:
        logits, _ = model(input_ids)
    finally:
        handle.remove()
    return logits, captured["h"]


@torch.no_grad()
def build_context_knn_datastore(model, tok, context: str):
    """Keys = h[:, :-1], values = token ids[:, 1:] from the context string alone."""
    if not context or not context.strip():
        return None
    ids = tok.encode(context, add_eos=False)
    if len(ids) < 2:
        return None
    # keep within ctx window
    max_t = model.max_seq_len
    if len(ids) > max_t:
        ids = ids[-max_t:]
    x = torch.tensor([ids], dtype=torch.long)
    _logits, h = forward_logits_and_hidden(model, x)  # h: [1, T, D]
    keys = h[0, :-1, :].float().contiguous()  # [T-1, D]
    vals = torch.tensor(ids[1:], dtype=torch.long)  # [T-1]
    # L2-normalize keys for cosine-ish knn (more stable on CPU than raw L2)
    keys = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return {"keys": keys, "vals": vals}


@torch.no_grad()
def generate_knn_lm(
    model,
    tok,
    prompt: str,
    context: str,
    max_new: int = 48,
    temperature: float = 0.0,
    top_k: int = 40,
    knn_k: int = 16,
    knn_lambda: float = 0.5,
    knn_temp: float = 0.1,
) -> str:
    """Interpolate LM next-token dist with passage-scoped kNN-LM.

    p = (1 - λ) · p_LM + λ · p_kNN
    p_kNN from softmax(-dist / knn_temp) over the k nearest context hidden states'
    successor tokens (classic datastore interpolation; Khandelwal et al.).
    """
    store = build_context_knn_datastore(model, tok, context)
    x = _encode_trim(tok, prompt, model.max_seq_len)
    if store is None:
        # fall back to plain greedy / sample
        for _ in range(max_new):
            ctx = x[:, -model.max_seq_len :]
            logits, _ = model(ctx)
            nxt = _pick(logits[:, -1, :], temperature, top_k)
            x = torch.cat([x, nxt], dim=1)
            if int(nxt.item()) == tok.eos_id:
                break
        return tok.decode(x[0].tolist())

    keys, vals = store["keys"], store["vals"]
    V = model.lm_head.weight.size(0)
    floor = 1e-12
    for _ in range(max_new):
        ctx = x[:, -model.max_seq_len :]
        logits, h = forward_logits_and_hidden(model, ctx)
        logits = logits[:, -1, :]  # [1, V]
        q = h[0, -1, :].float()
        q = q / q.norm().clamp_min(1e-8)
        # cosine distance = 1 - cos_sim; keys already unit
        sims = keys @ q  # [N]
        k_eff = min(int(knn_k), sims.numel())
        top_sim, top_idx = torch.topk(sims, k_eff)
        # convert similarity → soft neighbor weights
        # higher sim → lower distance
        dist = (1.0 - top_sim).clamp_min(0.0)
        weights = F.softmax(-dist / max(float(knn_temp), 1e-6), dim=0)
        neighbor_ids = vals[top_idx]

        # p_knn over vocab (sparse → dense)
        p_knn = torch.zeros(V, dtype=torch.float32)
        p_knn.scatter_add_(0, neighbor_ids, weights.float())
        p_knn = p_knn / p_knn.sum().clamp_min(floor)

        # p_lm
        if temperature is None or temperature <= 0:
            # near-greedy LM branch but still mix with knn
            lm_logits = logits / 1e-6
        else:
            lm_logits = logits / max(float(temperature), 1e-6)
        if top_k:
            v, _ = torch.topk(lm_logits, min(top_k, lm_logits.size(-1)))
            lm_logits = lm_logits.masked_fill(lm_logits < v[:, [-1]], float("-inf"))
        p_lm = F.softmax(lm_logits, dim=-1)[0].float()

        lam = float(knn_lambda)
        mix = (1.0 - lam) * p_lm + lam * p_knn
        mix = mix / mix.sum().clamp_min(floor)

        if temperature is None or temperature <= 0:
            nxt = mix.argmax().view(1, 1)
        else:
            nxt = torch.multinomial(mix, num_samples=1).view(1, 1)
        x = torch.cat([x, nxt], dim=1)
        if int(nxt.item()) == tok.eos_id:
            break
    return tok.decode(x[0].tolist())
