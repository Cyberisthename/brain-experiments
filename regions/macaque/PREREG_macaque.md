# Preregistration: MACAQUE area→neuron reservoir (+ FRANKEN graft)

**Locked:** 2026-10-29 07:39 America/New_York (EDT), BEFORE any macaque eval scores.  
**Dir:** `/workspace/flybrain/regions/macaque/`  
**Sources:** see `SOURCES.md` — primary wiring = Markov FLNe 29×29 (tract-tracing); OpenNeuro ds001875 MRI (~11.5 GB) not downloaded; TVB Zenodo dMRI weights secondary only / unused for FLN.

## Question

Does a ~2000-neuron reservoir expanded from real macaque **retrograde FLNe** (+ Dist_mm delays, hierarchy-proxy FF/FB) beat size-matched ER and area-FLN-shuffle controls on the locked composite, and does grafting that module onto FRANKEN beat an ER addon of the same size?

## Area → neuron expansion (locked rules)

1. **Area matrix:** Markov G29×29 FLNe among injected areas (`data/markov_fln29.npz`). Diagonal zeroed.  
2. **N:** target **2000** neurons (≈69 / area; exact N = 29 × n_per with n_per chosen so N∈[1980,2030]).  
3. **Within-area:** Dale E/I ≈ 80/20; sparse ER-like recurrent (p_intra=0.12); |w| ~ Exp(1) × sign(pre); no self-loops.  
4. **Inter-area:** for each FLNe>0 edge src→tgt, neuron-level edges with  
   `p = clip(0.08 * (FLN/FLN_max)^0.5, 0.002, 0.08)`;  
   |w| ∝ FLN_edge × Exp(1); sign = sign(pre). Cap total inter nnz for sanity.  
5. **Delays:** Dist_mm → `delay = 1 + round((d / d_max_obs) * (DMAX−1))`, DMAX=8; missing/zero dist → Uniform{1..DMAX}.  
6. **FF/FB (SLN proxy):** hierarchy ranks on the 29 areas (V1 lowest … prefrontal/cingulate highest). FF (src_rank < tgt_rank): prefer L2/3→L4 targeting; FB: prefer L5→L2/3. Flag `sln_proxy=hierarchy_rank` (measured SLN not in download). Layers per area: L2/3 35%, L4 25%, L5 25%, L6 15%.  
7. **Spectral:** block ρ=1 then global ρ=0.9.  
8. **Template seed:** graph expansion RNG **seed=11** frozen for MACAQUE adjacency; eval seeds only randomize Win / task IO / control rewires.

## Arms

| Arm | Definition |
|---|---|
| `MACAQUE` | real FLN expansion (frozen template) |
| `MACAQUE_ER` | ER match N/nnz/sign of MACAQUE; FIXED Uniform delays |
| `MACAQUE_SHUF` | shuffle **area** FLN preserving **row sums** (outgoing mass), re-expand with same rules + seed=11+seed_offset from eval seed; Dist kept |
| `DENS` | dens_match of MACAQUE (N/nnz, \|w\| bootstrap); FIXED delays |
| `FLY_CX` | real fly CX module |
| `FRANKEN` | mouse+human+fly_CX graft (existing frankenstein) |
| `FRANKEN+MACAQUE` | FRANKEN blocks + real MACAQUE module; bridges macaque L4/L5 ↔ fly like other cortex |
| `FRANKEN+ER_addon` | FRANKEN + ER block matched to MACAQUE N/nnz/sign; same bridge rule |

## Tasks / composite / α / seeds

- Tasks: **XOR_DELAY d=5** (1−NRMSE ↑), **NARMA-10** (NRMSE ↓), **MC** ↑, **PC_HARD** ↑ (cue=0.35, noise=1.5).  
- Leak=0.2; wash/tr/te=200/1500/500; ρ=0.9.  
- Ridge α **per arm × task** on seeds **90, 91, 92** only.  
- Composite: z-mean of XOR and −NARMA with μ/σ locked from **tune** pool (all arms), same as designed_brain / team_franken.  
- Eval seeds: **600–623** (n=24).

## Primary hypotheses (Bonferroni ×3, one-sided Wilcoxon on paired seed scores)

1. **H1:** MACAQUE composite > MACAQUE_ER composite  
2. **H2:** MACAQUE composite > MACAQUE_SHUF composite  
3. **H3:** FRANKEN+MACAQUE composite > FRANKEN+ER_addon composite  

**SUPPORTED** iff all three p_bonf = min(1, 3·p_raw) < 0.05 and mean inequalities hold. Else **NOT SUPPORTED**. Report each H pass/fail honestly.

## Ops

WORKERS=2; do **not** kill existing plasticity PIDs. Push code+prereg+results (no big MRI/data dumps > few MB curated CSVs/npz). Reservoir ≠ biology claim.
