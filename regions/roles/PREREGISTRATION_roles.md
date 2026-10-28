# Preregistration: Regions doing their real job
Written BEFORE any roles result. Queued after AetherLink lanes (or earlier if cores free once EMPN v2 is done).

## Motivation
Earlier region tests fed everything through the CX and used tasks that ignored each region's function. Hypothesis: real wiring + matching tasks will beat same-size random controls.

## Circuit note (locked before run)
The FlyWire extract used here (N=8765) has **no Kenyon cells** and **no LH neurons** (confirmed in `ann.tsv` against circuit roots). Therefore:
- **MB**: instantiate a biology-matched KC expansion (N_KC = 2000, ~7 random claws onto real ALPNs / PNs, kWTA ~5%), with **dopamine-gated KC→MBON plasticity** onto the **real 96 MBONs**. DAN broadcast uses real **PAM** (appetitive) and **PPL** (aversive) groups from FlyWire annotations.
- **LH**: a fixed innate PN→valence map (not plastic) with the same input dim as MB; tests whether innate valence speeds early odor learning when added to MB.
- **TuBu→CX**: real TuBu (150) driven by visual landmark features; projections into real EPG/PEN ring attractor cells for heading.

Controls for each: (i) **RAND** — same-size random-wired module, same inputs/plasticity/readout; (ii) **SHUF** — degree-preserving shuffle of the real region's edges.

## Tasks
### (a) MB — odor-reward association + reversal
- 4 synthetic odor patterns as sparse PN codes (overlap ~20%).
- Phase 1 (40 trials): odor A → reward (+1 via PAM), odor B → shock (−1 via PPL).
- Phase 2 (40 trials): reversal (A↔B contingencies).
- Metric: choice accuracy last-10 of each phase; trials-to-80% in phase 1; reversal recovery (accuracy last-10 phase 2).

### (b) LH — innate valence
- Same odors; LH provides fixed valence bias toward A (innately “good”).
- Arms: MB alone, MB+LH, RAND-MB+LH-fixed, SHUF-MB+LH.
- Metric: trials-to-80% in phase 1 (early learning speed).

### (c) TuBu → CX ring — heading / path integration
- Agent on a ring (36 bins); visual landmarks → TuBu; integrate heading.
- Task: after a random turn sequence, point home (heading error).
- Arms: real TuBu→EPG/PEN, RAND same-size, SHUF.

### (d) Combined fly in EMPN v2 maze
- CX + MB + LH + TuBu with real roles; maze gets odor cues tied to food.
- Compare to RANDMOD (v2) and to FLY (v2 learner, no role modules).
- Metrics: success, threat-under-loom, train-food rate (8 seeds).

## Seeds / stats
8 seeds (0–7). Bonferroni over the primary contrasts:
- MB-real > MB-RAND and MB-real > MB-SHUF on phase-1 last-10 and reversal last-10
- MB+LH < MB alone on trials-to-80% (faster)
- TuBu-real < TuBu-RAND on heading error
- Combined > RANDMOD on maze success (exploratory; may fail)

## Predictions
- R1: Real MB beats RAND and SHUF on acquisition and reversal.
- R2: LH speeds early acquisition (fewer trials to 80%).
- R3: Real TuBu→ring beats RAND/SHUF on heading error.
- R4: Combined fly ≥ RANDMOD on maze success when odor cues are present.
