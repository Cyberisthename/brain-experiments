# Preregistration: Episodic Meta-Plasticity Node (EMPN) on a FlyWire circuit in a gridworld
Written 2026-10-27 BEFORE any EMPN results were computed. EMPN design by the user; implementation choices below are mine.

## Circuit (real FlyWire v783 neuron IDs from annotations)
Included: CX (2,875), MBON (96), DAN (331: PAM 307, PPL 24), OA neurons (cell_type "OA-*", 43), descending neurons (super_class descending, 1,303), ALPN (685), TuBu (150), ORNs (cell_class olfactory, 2,282), and a fixed random sample of 1,000 of the 8,038 visual-projection neurons (VPN). Induced connectome subgraph, signed synapse counts, scaled to spectral radius ρ = 0.95, leak 0.5.
Skipped for size (noted): Kenyon cells (5,177), 7,038 of the VPNs, all optic-lobe neurons, the rest of the central brain.
Inputs: observation channels drive ORNs (odour) and the VPN sample (vision: walls, loom, door cue) via fixed random weights. Reward is injected into PAM; threat/punishment into PPL and OA (biologically motivated: PAM ~ reward, PPL1 ~ punishment, OA ~ arousal).
Motor: DN states (+ CX) are the readout features; 4 actions.

## Environment (9×9 maze; per seed a randomised layout family)
- Start bottom-left. Food at one of 2 locations; the location switches every 10-episode block and is signalled by odour ONLY in the first episode of each block (later episodes: no odour → must recall the path).
- Short route passes a threat zone; in each episode a looming threat is active with p = 0.5 and visible as a loom cue on approach. Entering the zone while active = −1 and episode ends ("risk").
- A shortcut door opens with p = 0.3 per episode (visual cue when adjacent); taking it saves steps. Long detour route is always safe.
- Reward +1 at food; −0.01 per step; max 60 steps. 200 training episodes then 60 evaluation episodes (learning continues; metrics on the last 60).

## Agents (all share the fly circuit and a linear Q-readout trained online by Q-learning, lr and ε identical)
- FLY: fly circuit alone (reactive).
- EMPN: fly + three parts:
  - TBU: hippocampus module (EC/DG kWTA/CA3/CA1 as in ../hippo) WITH Hebbian CA3 storage: snippets (DG code of position-observation at t → t+1) written into CA3 hetero-associative weights; recall = iterate CA3 from the current cue to predict the next stored step.
  - VTM: reads PAM vs PPL/OA activity of the fly circuit at episode end; tags the episode's snippets with valence. Positive episodes are stored in TBU; threat-hit cells are stored in a negative risk map; neutral episodes not stored.
  - CFS: learned world model (transition counts from experience) + forward simulation of each candidate action for H = 6 steps; scores = recalled-path/food value − risk map × (loom cue active). Its action scores are injected into CX → DN (added to the DN-readout Q-values, weight β).
- RANDMOD: fly + a random recurrent module with the same parameter count as EMPN's TBU, its state appended to Q features (no Hebbian storage, no planning).
- Ablations: EMPN−TBU (no recall; CFS plans only toward odour), EMPN−VTM (no valence gating: every episode stored, no risk map), EMPN−CFS (TBU recall used as a 1-step bias, no forward simulation).
- 8 seeds per agent.

## Metrics (last 60 episodes)
success rate (food reached, no threat hit); steps to food (successful episodes); risk decisions: % of loom-active episodes entering the threat zone (bad) and % of loom-inactive episodes using the short route (good); success in no-odour episodes (memory).
Stats: Mann–Whitney EMPN vs each other agent per metric (n=8 vs 8); Bonferroni over all those tests.

## Honest caveat stated in advance
EMPN's CFS and TBU are hand-designed algorithms (planning + associative memory) bolted onto a fixed fly circuit. If EMPN wins, that shows these algorithms help in this task, not that the fly connectome computes them.

## Predictions
- E1: EMPN beats FLY and RANDMOD on success rate, mainly in no-odour episodes.
- E2: RANDMOD ≈ FLY.
- E3: Removing TBU hurts no-odour success most; removing VTM raises threat hits; removing CFS raises steps to food (fewer shortcut/detour uses).
- E4: FLY alone near chance on no-odour episodes (it has no cross-episode memory).

## Deviations logged BEFORE the full run (after smoke tests on seed 0 only)
- Grid shrunk 9x9 -> 7x7, max steps 60 -> 80: a random walk found food only 1% of the time on 9x9 (11.5% on 7x7), so no agent could learn.
- Odour cue on for first 3 episodes of each 10-episode block (was 1).
- Exploration epsilon decays 0.3 -> 0.05 over training (was fixed 0.1).
- Fly readout: 6 settle steps (was 2), input gain x3, DN features z-scored using a 150-sample calibration at init. Reason: with the original setting DN activity barely changed with input (mean std 0.0005), i.e. the readout was blind.
- Smoke (seed 0): FLY success 0.00, EMPN 0.30. These seed-0 numbers are reported but the full run uses all 8 seeds.

## Addendum (user spec update, before the full run)
- The TBU is upgraded to the plastic hippocampus (regions/hippo_plastic):
  - CA3 covariance+BCM Hebbian on recurrents and mossy fibres.
  - CA1 contrastive delta.
  - CA1→E-PG/P-EG targets.
- Extra EMPN arms: TBU plasticity off, and CA1 targets at FB tangential or random CX.
- Plastic TBU smoke (seed 0, before the full run): EMPN 0.05 success, TBU-static 0.02. A try with decay toward baseline gave 0.00 and was reverted. Design frozen now. No more tuning on seed 0, to avoid fitting to test data. Recall check: after storing a 9-step path, the plastic CA3 replays it perfectly.

## Addendum 2: SCNC (user spec, logged BEFORE any SCNC result; only a 30-episode crash test was run)
- SCNC (500 neurons). Layer 1 is 400 fixed sparse random tanh units (ELM-style). It takes maze observations and feeds a 200-unit EC code to the plastic hippocampus (replacing the place one-hot).
- Layer 2 is 100 random tanh units. They take time since food, distance to the threat zone (when looming), food events, and threat events.
  - Output: scalar V = |c·h − baseline|, normalised. V is about 0.002–0.01 normally and 0.7–0.85 at food or danger.
- V gates the CA3/CA1 learning rate as a three-factor rule (η·eligibility). The eligibility trace decays 0.9 per step backward, clipped at 3. With SCNC, every episode is stored, weighted by V. This replaces the episode-end VTM food gate. Threat episodes still update the risk map.
- Arms:
  - EMPN+SCNC.
  - EMPN+SCNC-Vmean: V fixed at its running mean.
  - EMPN+SCNC-Vshuf: V drawn as a random permutation of past V values, so the timing is broken.
  - EMPN+RAND500: 500 random recurrent units added to the readout.
- Neuron count: the circuit is 8,765 neurons, not the 6,375 the user estimated. Adding SCNC brings it to 9,265.
- **Prediction S1:** SCNC > Vmean and Vshuf on success rate (the gating timing matters). SCNC > RAND500.
- **Honest prior:** I expect S1 to probably fail or be small. That's because SCNC also stores danger paths in the same goal memory, and earlier smoke tests showed EMPN gains are fragile. Same 8 seeds. Bonferroni over 3 comparisons (SCNC vs Vmean, Vshuf, RAND500).
- Resources: the user lifted the limits. The run now uses 7 workers, 1 thread each, no nice. The design is unchanged.

## Addendum 3: CTB, MGM, CVB organs (logged BEFORE any result; only a 20-episode crash test on seed 0 was run)
- **Bug fix applied to all arms before the full run:** the agent's observation `o` was never refreshed inside an episode. So the CFS loom flag was stale (it was always the flag at the start position). It now updates every step.
- **CTB (250 gating units).** At junctions (3 or more open moves), it pauses with no environment step taken. For each action it rolls out 3 steps: the world-model next cell, then 2 CA3 recall steps.
  - Score = remembered-goal value − (loom × risk on the path) − 0.5 × dead-end flag.
  - With CFS present, the bias = 0.5·CFS + CTB.
  - It does not model the CX heading explicitly. The direction is implicit in the action, which is a simplification.
- **MGM (100 units).** DG k drops to 10/2000 (0.5%) in familiar states. It rises to 100/2000 (5%) when the state has been visited at most once, or when SCNC V > 0.3. The default without MGM is 40 (2%).
  - Clutter is measured as the relative L1 change of CA3 recurrent weights from baseline, plus the fraction of CA3 rows that changed.
- **CVB (150 units).** curiosity = (no loom) × (V low); fear = loom-near; urgency = time since food / max steps.
  - Action bias = curiosity·(1−urgency)·novelty of the next cell, plus 2·fear·habit (the most-used route from this cell).
- **ALL3RAND.** Each organ is replaced by a random tanh network of the same size, with outputs at a matched scale.
  - CTB → random action scores from the observation.
  - MGM → a random high/low k gate.
  - CVB → a random action bias from (fear, curiosity, urgency).
- **Arms:** EMPN+SCNC (base), +CTB, +MGM, +CVB, +ALL3, +ALL3RAND. Same 8 seeds.
  - Bonferroni over 5 comparisons, each vs the base: CTB, MGM, CVB, ALL3, and ALL3 vs ALL3RAND.
  - CTB vs plain CFS is the base arm's comparison (the base has CFS only).
- **Predictions:**
  - O1: CTB ≥ base on success and lowers threat hits under loom.
  - O2: MGM lowers CA3 clutter vs base but does not change success.
  - O3: CVB raises the shortcut rate when safe and lowers it under loom.
  - O4: ALL3 > ALL3RAND.
- **Honest prior:** after the fragile smoke results I expect most differences not to survive correction.
