# NEURON-1 results — 2026-11-05 ET (synthetic memory stream, 2,000 episodes, 5 seeds, no QPU, no external models)
Files: PREREG.md (incl. Amendment 1), run_neuron1.py (frozen), run_amend1_cam_mps.py, results/{rows,summary,amend1_rows}.json, logs.

| System | Stored bytes | Bytes / correct fact | Exact | Retention | Relation | 2-hop | Ambiguous | False-memory |
|---|---|---|---|---|---|---|---|---|
| 1 zstd + grep | 20.9 KB | **11.7** | 100% | 100% | 100% | 100% | 100% | 0% |
| 2 TF-IDF/SVD RAG | 1.43 MB | 800 | 100% | 99.9% | 100% | 100% | 93.4% | 1.1% |
| 3 JARVIS pack router | 52.3 KB | 29.2 | 100% | 100% | 100% | 100% | 100% | 0% |
| 4 HDC 10k | 2.50 MB | 1,397 | 100% | 100% | 93.4% | 93.5% | 100% | 0% |
| 5 NEURON hybrid | 22.4 KB | 12.5 | 100% | 100% | 100% | 100% | 100% | 0% (0 reconstruction errors) |
| 6 CAM | 29.4 KB | 16.4 | 100% | 100% | 100% | 100% | 100% | 0% |
| 6 CAM static coords | 29.3 KB | 16.4 | same as CAM | | | | | 0% |
| 6 CAM no spreading | 29.3 KB | 16.4 | 100% | 100% | 100% | **0%** | 100% | 0% |
| 7 MPS chi=4/16/64 (exploratory) | 4/16/89 KB | — | 0% / 0% / 0.07% | | | | | |

Facts / (storage + retrieval bytes): NEURON 0.063, CAM 0.047 (static 0.047, nospread 0.039), grep 0.0067, RAG 0.0008.

H1 (NEURON beats RAG & HDC on bytes/fact, no worse false memory): SUPPORTED, 5/5 seeds — but trivially (RAG/HDC pay for dense vectors).
Secondary: NEURON does NOT beat plain zstd+grep (0/5 seeds; 12.5 vs 11.7 B/fact).
H2 (CAM >= NEURON on facts/(storage+retrieval)): NOT SUPPORTED, 0/5 seeds.
Ablations: spreading activation is what gives CAM 2-hop (100% -> 0% without it). Moving coordinates did nothing
(static == full); the sandbox kept 0 changes in all seeds. MPS can't hold exact values (null).
Caveats: benchmark is templated and hit a ceiling for structured systems; the shared regex reader does the heavy lifting.
Process notes: run 1 crashed pre-results (uint16 day overflow -> int32); the first full run was interrupted after seeds 0-2
and rerun from scratch with identical code; Amendment 1 written after seeing those partial ceiling results; amend script
had a JSON-serialization crash after all rows were saved — aggregated from saved rows, then fixed.
Next: real messy records (/workspace/ibmq RESULTS.md, chat logs) with paraphrased queries, where exact lookup can't win for free.
