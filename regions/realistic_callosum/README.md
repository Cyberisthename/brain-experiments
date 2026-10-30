# realistic_callosum

Follow-up to `oscillation_gating` callosum arms. Uses Griffa70 / Rosen–Halgren-style real interhemispheric priors (mostly heterotopic, strong minority homotopic, FC signs, myelin delays) with **left-only readout** so XOR is bridge-required.

- Prereg: `PREREG_realistic_callosum.md`
- Data notes (local): `../../connectomes/human/griffa70/NOTES_callosum_data.md`
- Run: `python3 realistic_callosum.py --eval --workers 2`
- Analyze: `python3 analyze_realistic_callosum.py`
