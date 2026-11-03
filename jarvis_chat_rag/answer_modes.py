"""Unified answer helpers for training-free RAG modes."""
from __future__ import annotations

from typing import Optional

from decode_extras import (
    generate_cad,
    generate_copy_bias,
    generate_knn_lm,
    generate_soft_exclusion,
    make_copy_prompt,
)
from extractive import best_extractive, format_hybrid
from model_io import continuation_only, generate
from dictionary_lookup import answer_define, extract_define_query


def make_prompt(question: str) -> str:
    q = question.strip()
    if q.lower().startswith("question:"):
        return q if "answer:" in q.lower() else q + "\nAnswer:"
    return f"Question: {q}\nAnswer:"


def answer_one_mode(
    model,
    tok,
    retriever,
    question: str,
    mode: str,
    max_new: int = 32,
    temperature: float = 0.0,
    min_ext: float = 1.5,
    cad_alpha: float = 1.0,
    copy_bias: float = 3.0,
    prefer_dict: bool = True,
    extractive_only: bool = False,
    soft_strength: float = 0.7,
    knn_lambda: float = 0.5,
    knn_k: int = 16,
) -> dict:
    """mode in {gen, hybrid, cad, copybias, cad_hybrid, dict, soft_exclusion, knn}.

    Soft-exclusion = negative-constraint decoding (user idea).
    knn = passage-scoped kNN-LM interpolation.
    """
    mode = mode.lower().strip()
    hits = retriever.search(question, k=retriever.k)
    ctx = retriever.context(question)
    prompt = make_prompt(question)
    aug = retriever.augment(prompt, query=question)

    dict_ans = answer_define(question) if prefer_dict else None
    if mode == "dict" or (
        prefer_dict
        and dict_ans
        and mode in {"hybrid", "cad_hybrid"}
        and extract_define_query(question)
    ):
        if mode == "dict" or extract_define_query(question):
            return {
                "mode": "dict",
                "question": question,
                "context_preview": ctx[:500],
                "prompt_preview": aug[:500],
                "generation": "",
                "extractive": {
                    "span": dict_ans.split("\n")[1] if dict_ans and "\n" in dict_ans else (dict_ans or ""),
                    "sentence": dict_ans or "",
                    "source": "wordnet",
                    "score": 99.0,
                },
                "display": dict_ans or "[DICT] miss",
                "dict": dict_ans,
            }

    full = ""
    cont = ""
    if mode in {"gen", "hybrid"}:
        if mode == "hybrid" and extractive_only:
            full, cont = "", ""
        else:
            full = generate(model, tok, aug, max_new=max_new, temperature=temperature)
            cont = continuation_only(full, aug).strip()
    elif mode == "copybias":
        copy_prompt = make_copy_prompt(question)
        aug_copy = retriever.augment(copy_prompt, query=question)
        full = generate_copy_bias(
            model, tok, aug_copy, ctx, max_new=max_new, temperature=temperature, bias=copy_bias
        )
        cont = continuation_only(full, aug_copy).strip()
    elif mode in {"cad", "cad_hybrid"}:
        full = generate_cad(
            model,
            tok,
            prompt_with_ctx=aug,
            prompt_no_ctx=prompt,
            max_new=max_new,
            temperature=temperature,
            alpha=cad_alpha,
        )
        cont = continuation_only(full, aug).strip()
    elif mode in {"soft_exclusion", "softexcl", "soft"}:
        full = generate_soft_exclusion(
            model,
            tok,
            aug,
            ctx,
            max_new=max_new,
            # greedy soft-exclusion for reproducibility in eval
            temperature=0.0 if temperature is None or temperature <= 0 else temperature,
            top_k=40,
            strength=soft_strength,
            unsupported_strength=soft_strength,
        )
        cont = continuation_only(full, aug).strip()
    elif mode in {"knn", "knn_lm"}:
        full = generate_knn_lm(
            model,
            tok,
            aug,
            ctx,
            max_new=max_new,
            temperature=temperature,
            knn_k=knn_k,
            knn_lambda=knn_lambda,
        )
        cont = continuation_only(full, aug).strip()
    else:
        raise ValueError(f"unknown mode {mode}")

    ext = best_extractive(question, hits, min_score=min_ext)
    if mode in {"hybrid", "cad_hybrid"}:
        display = format_hybrid(ext, cont, mode_label=mode)
    else:
        display = cont

    return {
        "mode": mode,
        "question": question,
        "context_preview": ctx[:500],
        "prompt_preview": aug[:500],
        "generation": cont,
        "full": full,
        "extractive": ext,
        "display": display,
        "dict": dict_ans,
    }
