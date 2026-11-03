# Y/Z/G Cognitive Router — results snapshot

Copied from local `run_log.txt` (benchmark may still be running).

```
loaded 120 questions
Loading BM25 index…
chunks=31328 rss=714.0MB
split train=84 test=36
Tuning on train…
  tune graph_ppr: {'damping': 0.8} train_pack=0.083
  tune yz: {'lam': 0.7, 'z_mix': 0.2} train_pack=0.369
  tune yzg: {'lam': 0.85, 'z_mix': 0.35, 'g_mix': 0.4} train_pack=0.381
  tune yz_zrand: {'lam': 0.85, 'z_mix': 0.35} train_pack=0.369
  tune softmax: {'temperature': 0.5} train_pack=0.369
  tune existing: {} train_pack=0.714

=== seed 0 ===
  existing: pack=0.583 R@3=0.583 hyb=0.583 lat=83.30ms auroc_p=0.095
  graph_ppr: pack=0.083 R@3=0.333 hyb=0.333 lat=97.41ms auroc_p=0.535
  yz: pack=0.250 R@3=0.472 hyb=0.472 lat=84.72ms auroc_p=0.259
  yzg: pack=0.167 R@3=0.417 hyb=0.417 lat=97.76ms auroc_p=0.294
  yz_zrand: pack=0.250 R@3=0.472 hyb=0.472 lat=98.45ms auroc_p=0.255
  softmax: pack=0.250 R@3=0.472 hyb=0.472 lat=103.87ms auroc_p=0.280

=== seed 1 ===
  existing: pack=0.583 R@3=0.583 hyb=0.583 lat=97.14ms auroc_p=0.095
  graph_ppr: pack=0.083 R@3=0.333 hyb=0.333 lat=115.49ms auroc_p=0.535
  yz: pack=0.250 R@3=0.472 hyb=0.472 lat=89.02ms auroc_p=0.255
  yzg: pack=0.222 R@3=0.444 hyb=0.444 lat=93.02ms auroc_p=0.205

```
