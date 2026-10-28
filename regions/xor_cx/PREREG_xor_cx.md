# Preregistration: FLY_CX alone on XOR_DELAY (own matched controls)

**Locked:** 2026-10-28 16:23 America/New_York (EDT), BEFORE any `xor_cx` eval scores.  
**Parent:** `regions/xor_rep` — FRANKEN > ER/SHUF on XOR_DELAY d=5 SUPPORTED, but mechanism showed **fly_CX carries the effect** (FLY_CX alone 0.63 > FRANKEN 0.48; lesion fly kills; lesion human helps).  
**Dir:** `/workspace/flybrain/regions/xor_cx/`  
**Question:** Does **real FlyWire CX wiring** beat size-/density-/degree-matched **random** controls of the *same* CX module on delayed XOR?

## Recipe (frozen; XOR task bit-exact with xor_rep / trained_bridges)

- Network: **FLY_CX alone** (`fly_cx_module.npz`, N=2875). No mouse/human graft.
- Delays: **FIXED** Uniform{1..dmax}, dmax=5 (same as xor_rep FLY_CX under REALMYELIN mode when no human block — no myelin map applies).
- Leak=0.2. Drive = CX L4 ports (`frankenstein.layer_sets`).
- XOR: identical `xor_delay_score` (2-input delayed XOR; y=±1; metric 1−NRMSE; wash/tr/te=200/1500/500).
- Ridge α **per arm** on seeds **90, 91, 92** only (maximize mean XOR d=5 FIXED; α grid `{1e-8,1e-6,1e-4,1e-2,0.1,1,10}`; never peek at 300–323). Lock `out/alpha_choice.json` before eval.
- Eval seeds: **300–323** (24 NEW; never 0–7, 90–92, 100–123, 200–223).
- Block/module scaled to ρ=1 then global ρ=0.9 (same as `frankenstein.build_arm("FLY_CX")`).
- `fastcore2` opt-in if importable.

## Arms

| Arm | Definition |
|---|---|
| `CX` | real fly_CX |
| `ER` | ER matched N / nnz / sign (`er_match`) |
| `SHUF` | degree-preserving shuffle (`degree_shuffle`) |
| `DENS` | dens_match (N/nnz + \|w\| bootstrap) |
| `EI_SHUF` | **E/I-class shuffle**: rewire edges only within (sign_pre × sign_post) classes; preserves E/I edge-type counts and out-sign rule |
| `SPEC` | **secondary**: dens_match then non-uniform temper (shrink smallest-magnitude half of edges, rescale ρ=1) until top-64 \|λ\| fraction >0.5 matches real CX ±0.05 (or 25 iters); tests whether spectral bulk alone explains XOR |

Primary family: **CX, ER, SHUF, DENS** only. EI_SHUF and SPEC are secondary / sanity.

## Primary hypothesis (locked)

- **H1:** On XOR_DELAY **d=5**, FIXED delays:  
  `mean(CX) > mean(ER)` **and** `mean(CX) > mean(SHUF)` **and** `mean(CX) > mean(DENS)`.
- Test: paired Wilcoxon signed-rank **one-sided** (greater), n=24, same seed paired.
- **Bonferroni ×3** over the three primary contrasts.
- **SUPPORTED** iff all three have p_bonf < 0.05 **and** mean(CX) exceeds each control.
- Else **NOT SUPPORTED**.

## Secondary (report; do not expand primary Bonferroni)

1. XOR_DELAY at **d ∈ {2, 3, 8, 10}** for CX / ER / SHUF / DENS.
2. EI_SHUF at d=5 (structure within E/I classes).
3. SPEC at d=5 (spectral-bulk-matched random).
4. Effect sizes: paired Cohen’s d_z = mean(Δ)/sd(Δ).
5. Report top-64 frac(\|λ\|>0.5) per arm (mean over tune seeds) as a spectral audit.

## Ops / honesty

- WORKERS≤2, nice, nohup CPU. Do not kill unrelated jobs / Kaggle sessions.
- Push code+prereg+results to `Cyberisthename/brain-experiments` (no big data, no secrets).
- This is a reservoir metric, not a claim that the fly “computes XOR” biologically.
- Report negatives. If CX fails vs DENS/ER, say so — parent mechanism suggested CX is strong, but that was without CX-matched controls.
