# Figures

PNG figures for key runs live on the experiment machine under `/workspace/flybrain/figs/` (and some under `regions/*/out/`). The GitHub connector used for this repo is **text-only**, so binary `.png` files are not committed here.

| Figure (local) | What it shows |
|---|---|
| `compass_heading.png` | Compass / path-integration heading errors by arm |
| `fewshot_delay_mc_curves.png` | Few-shot MC learning curves with/without delays |
| `celegans_effect_by_worm.png` | C. elegans real-vs-random effect by individual worm |
| `franken_diagram.png` / `franken_matrix.png` / `franken_eigs.png` | Frankenstein layout / matrix / eigenspectrum |

Regenerate frankenstein figs with `_make_franken_figs.py`. Meta: `franken_figs_meta.json`.
