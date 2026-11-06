# NEURON-1 preregistration (written before any run) — 2026-11-05 ET

No external models, no QPU. Embeddings = TF-IDF + truncated SVD (numpy, from scratch).
Frozen script: `run_neuron1.py` (sha256 recorded in results/run_meta.json). Seeds: 0,1,2,3,4.

## Data (per seed, synthetic)
Phase 1: 1,000 episodes; Phase 2: +1,000 new episodes. Types: measurement
(entity, device, quantity, value, unit, date, source, recal_since_prev), coupling relation
(entity A coupled to B, ZZ kHz, date, source), calibration event. Distractors: near-duplicate
entity names (Q45/Q54/Q45b), same entity+quantity on adjacent dates, same value on other entities.

## Queries (per seed): exact (dated value), relation (who is coupled to X), 2-hop
(quantity of the partner of X on date), ambiguous (undated value when >1 value exists;
correct = abstain/flag ambiguous or return the full set; a single value = false memory),
retention (phase-1 exact queries re-asked after phase 2).

## Systems
1 zstd raw + exact grep; 2 TF-IDF/SVD RAG (k=5); 3 JARVIS pack router (BM25 from
jarvis_kaggle memory_rag + keyword pack routing, as in chat_rag/pack_router); 4 HDC bipolar
D=10,000 role-filler bind/bundle + cleanup memory; 5 NEURON hybrid (A exact compressed
episodic store of schema residuals, B schema/relation graph, C per-cluster index adapters;
answers must cite an A row). Systems 1-3 share one regex reader (same ambiguity rule).

## Metrics
bytes/correct fact = stored bytes / (exact recall x #facts); exact recall; relation; 2-hop;
ambiguous accuracy; retention; false-memory rate = confident wrong answers / all queries.

## Primary hypothesis H1
NEURON has lower bytes/correct-fact than BOTH RAG and HDC in >=4/5 seeds AND mean
false-memory rate <= min(RAG, HDC) + 1 pp. Otherwise H1 NOT SUPPORTED.
Secondary (reported, not hypothesis): NEURON vs zstd+grep on bytes/fact.
Caveat preregistered: data is synthetic and templated; favours structured systems.

## AMENDMENT 1 — 2026-11-05 08:55 ET (added after a partial first run of systems 1-5 on seeds 0-2; the main script is unchanged except a pre-result int overflow fix uint16->int32 for day offsets)
Partial seeds already showed a ceiling (grep/router/NEURON at 100%). Arms below are separate script
`run_amend1_cam_mps.py` (same data generator/queries/seeds 0-4, imported from the frozen script).
6. CAM (Coordinate Associative Memory): per-entity cluster coordinate (12-d, init from TF-IDF/SVD of its
records), weighted links (L2 from relation records), spreading activation (1 step, decay 0.5),
append-only layers L0 token index / L1 originals (phase 1) / L2 relations / L3 new (phase 2), versions never
overwritten. Online update on a separate 300-query training stream: pull co-activated coords together (lr 0.05),
+0.1 link weight. Universe Sandbox: copy links/coords, try pruning links below a threshold; keep only if a
held-out 200-query validation score improves; originals untouched. Answers read by the shared reader from L1/L3 originals.
Ablations: CAM-static (no online update, no sandbox), CAM-nospread (no spreading activation; 2-hop answered in ONE retrieval pass).
CAM metric: correct facts / (storage bytes + mean retrieval bytes read per query); reported for all systems.
CAM hypothesis H2: CAM >= NEURON on this metric with false-memory <= NEURON + 1pp, >=4/5 seeds.
Ablation predictions: nospread loses 2-hop; static vs full shows whether moving coordinates matter.
7. Exploratory (no hypothesis): MPS/tensor-train encoding of the entity x quantity x day value tensor,
bond dims 4/16/64, fp16; correct if rounded reconstruction string == stored value. Classical simulation only. No QPU.
