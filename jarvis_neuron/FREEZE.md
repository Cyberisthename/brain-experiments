# NEURON freeze manifest — immutable baselines
**When (America/New_York):** 2026-11-05 10:22 AM UTC-05:00
**Python:** 3.13.5 · **OS:** Linux-6.12.94+-x86_64-with-glibc2.41

## Policy
- NEURON-1 archive: `frozen/neuron1_v1` (excludes `neuron2/`).
- NEURON-2 archive: `frozen/neuron2_v1`.
- Archives are `chmod a-w`. **Do not edit originals or archives.**
- `summary_corrected.json` is a post-hoc correction; originals stay as-is.

## Config / seeds
- NEURON-2 seeds: `[0, 1, 2, 3, 4]`
- SVD_DIM=128, TOP_K=12, CAM_DIM=12
- Script sha256 **at original run:** `5f3beb4880b9bca0d4c157761941cd9fd60dd3a9ab0c2aef03646636be318cae`
- Script sha256 **on disk at freeze:** `454ff7fa5fddf0d6cd872446e722b9aa3b2607a40defc54d2c580c8192fb2caa`
- Reconstruction of run script: **verified** (`5f3beb4880b9bca0d4c157761941cd9fd60dd3a9ab0c2aef03646636be318cae`).

## summary_corrected.json provenance
- **Who:** post-run agent correction (not `run_neuron2.py`)
- **When (America/New_York):** 2026-11-05 09:49:27 UTC-05:00
- **Why:** pad-accounting bug made `stored_bytes_native` negative; correction sets native = capped − pad.
- **sha256:** `1cb2b3006efd3fe1b501adaf56c3c8b9aa93a03042fbf528f24c9f2ca0f9208f`
- Original `summary.json` mtime ET: 2026-11-05 09:48:27 UTC-05:00
- Script edit mtime ET: 2026-11-05 09:49:32 UTC-05:00

## NEURON-1 files
Archive `/workspace/jarvis/neuron/frozen/neuron1_v1` — 10 files

| path | bytes | sha256 |
|---|---:|---|
| `PREREG.md` | 3644 | `a058c7a124120ad7f08a41dc90cc49c51816a8edcb7ce3c650d0983cab5d9d39` |
| `RESULTS.md` | 2342 | `1f49285deb315a75f47dda71b6d5519cb8683a7b0b72128715abda0121605205` |
| `results/amend1_log.txt` | 1055 | `7fd604ceb563db412f97201089167060b81f5e3059874f0e46ededdd189a8f27` |
| `results/amend1_rows.json` | 9622 | `0bb1ee64416b2104f5f6642ed5194ffd2cfce938cfb310f6ff0d0e765940425e` |
| `results/log.txt` | 8240 | `4ef5a174bd74d4307c0e2d64a8f1719b0000f1b1ab929f68819207c68890e112` |
| `results/rows.json` | 7301 | `25448e6ccfada55bfbd028f6a7cd385af3b20d8ecf0847fb0aad114ef489ffc1` |
| `results/run_meta.json` | 149 | `5b4e7289203f6fddc777ba51203ecda9f67c1bbe2fcadffb69c43e24ef16a03b` |
| `results/summary.json` | 2132 | `449a382756c7e1060b0f0ac1081ced1bed9a5460e1a2e959eae776ea1f72d4cb` |
| `run_amend1_cam_mps.py` | 7963 | `31c5a364cd5689a26fa001f67b8d7cfacd1f858a705d9eec596ca304c3641641` |
| `run_neuron1.py` | 17477 | `b65d9d228c7ebca319bdbb707b6657ebe0cf4a550a480bec837d29476103a3a9` |

## NEURON-2 files
Archive `/workspace/jarvis/neuron/frozen/neuron2_v1` — 12 files

| path | bytes | sha256 |
|---|---:|---|
| `PREREG.md` | 6010 | `aafb36750543e6aea2aaee90a97e74b9085480345a80d42e8b97deeee255c842` |
| `RESULTS.md` | 3228 | `d1a5d466ed35783c9d32ca3a080de17fd4a1db445cc2875bb12c9d62a0a9045e` |
| `results/corpus_meta.json` | 10542 | `87ef2ebf61977baf14fdde4d4f43bcb0868c2312a8669433ef1d84450bf57b82` |
| `results/facts.json` | 9461 | `7c0b6d4b8d2ea4edebc2c9cd4b27dce471d567c6b6a3d3d3461cbc9aaed9cd42` |
| `results/interference.json` | 9460 | `8e059b941b2c5324f9f97bc1382ada6c7813a4a1027c4fe861bdb035a99e45f5` |
| `results/lex_overlap.json` | 611 | `3821d272026de2ee917592bcf27d6235215aa5c8c54e2a3749ae5198b2ebbff6` |
| `results/rows.json` | 34233 | `fd25abfad6766cc6120437425113ff01abf2d116d3e4644958aca7e1c93b7459` |
| `results/run_log.txt` | 9982 | `f73bd2a9d59953044c991cd767c4f87974df2bfc72832831558e891453905920` |
| `results/run_meta.json` | 245 | `2af0d79d1a7093762b7456818638d01f29e325bec519f7877172c9fb59b18623` |
| `results/summary.json` | 19150 | `02d697dc435bff6900904ba1ce2c44c1bf2da1df4060c50524b2eeb8a4c70fe5` |
| `results/summary_corrected.json` | 2924 | `1cb2b3006efd3fe1b501adaf56c3c8b9aa93a03042fbf528f24c9f2ca0f9208f` |
| `run_neuron2.py` | 67393 | `454ff7fa5fddf0d6cd872446e722b9aa3b2607a40defc54d2c580c8192fb2caa` |

## pip freeze
```
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
annotated-doc==0.0.5
anyio==4.15.1
attrs==26.1.0
bcrypt==4.2.0
beautifulsoup4==4.15.0
certifi==2026.10.23
cffi==2.1.1
chardet==5.2.0
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
cobble==0.1.4
contourpy==1.4.0
cryptography==43.0.0
cycler==0.12.1
datasets==5.1.0
dbus_next==0.2.3
decorator==5.3.1
defusedxml==0.7.1
Deprecated==1.2.18
dill==0.4.1
et_xmlfile==2.0.0
filelock==3.32.3
flatbuffers==25.12.19
fonttools==4.65.0
frozenlist==1.8.0
fsspec==2026.7.0
h11==0.16.0
h5py==3.16.0
hf-xet==1.7.1
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.3.0
idna==3.20
iniconfig==2.3.1
# Editable install with no version control (jarvis-core==0.1.0)
-e /workspace/jarvis_model
Jinja2==3.1.6
joblib==1.6.0
jwcrypto==1.5.6
kiwisolver==1.5.1
llvmlite==0.50.0
lxml==6.1.3
magika==0.6.3
mammoth==1.11.0
markdown-it-py==4.2.0
markdownify==1.2.3
markitdown==0.1.7
MarkupSafe==3.0.3
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
multidict==7.2.1
multiprocess==0.70.19
networkx==3.6.1
nltk==3.10.4
numba==0.68.1
numpy==2.5.4
obspy==1.5.1
onnxruntime==1.31.1
openpyxl==3.1.5
packaging==26.3
pandas==3.0.6
pillow==13.0.0
pluggy==1.7.0
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
py-spy==0.4.2
pyarrow==26.0.0
pycparser==3.12
Pygments==2.21.0
PyGObject==3.50.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==3.0.0
python-docx==1.2.0
python-dotenv==1.2.4
python-pptx==1.0.2
PyWavelets==1.10.0
PyYAML==6.0.3
rank-bm25==0.2.2
redis==6.1.0
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scipy==1.18.1
sentencepiece==0.2.2
setuptools==78.1.0
shellingham==1.5.4
six==1.17.0
soundfile==0.14.0
soupsieve==3.0.4
SQLAlchemy==2.1.8
sympy==1.14.0
tabulate==0.10.0
tokenizers==0.23.3
torch==2.14.1+cpu
tqdm==4.70.1
transformers==5.22.0
truststore==0.10.5
typer==0.27.3
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
websockify==0.12.0
wheel==0.46.1
wrapt==1.15.0
xlsxwriter==3.3.0
xxhash==4.0.2
yarl==1.25.1
yt-dlp==2026.10.21
zstandard==0.25.0
```

## Raw env capture
```
=== date ===
Thu Nov  5 15:22:48 UTC 2026
=== uname ===
Linux grok-bot-vm-302353508 6.12.94+ #1 SMP PREEMPT_DYNAMIC Mon Oct 26 15:07:58 UTC 2026 x86_64 GNU/Linux
=== python ===
Python 3.13.5
/usr/bin/python3
=== pip freeze ===
accelerate==1.15.0
aiohappyeyeballs==2.7.1
aiohttp==3.14.5
aiosignal==1.4.0
annotated-doc==0.0.5
anyio==4.15.1
attrs==26.1.0
bcrypt==4.2.0
beautifulsoup4==4.15.0
certifi==2026.10.23
cffi==2.1.1
chardet==5.2.0
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
cobble==0.1.4
contourpy==1.4.0
cryptography==43.0.0
cycler==0.12.1
datasets==5.1.0
dbus_next==0.2.3
decorator==5.3.1
defusedxml==0.7.1
Deprecated==1.2.18
dill==0.4.1
et_xmlfile==2.0.0
filelock==3.32.3
flatbuffers==25.12.19
fonttools==4.65.0
frozenlist==1.8.0
fsspec==2026.7.0
h11==0.16.0
h5py==3.16.0
hf-xet==1.7.1
httpcore2==2.16.0
httpx2==2.16.0
huggingface_hub==2.3.0
idna==3.20
iniconfig==2.3.1
# Editable install with no version control (jarvis-core==0.1.0)
-e /workspace/jarvis_model
Jinja2==3.1.6
joblib==1.6.0
jwcrypto==1.5.6
kiwisolver==1.5.1
llvmlite==0.50.0
lxml==6.1.3
magika==0.6.3
mammoth==1.11.0
markdown-it-py==4.2.0
markdownify==1.2.3
markitdown==0.1.7
MarkupSafe==3.0.3
matplotlib==3.11.2
mdurl==0.1.2
mpmath==1.3.0
multidict==7.2.1
multiprocess==0.70.19
networkx==3.6.1
nltk==3.10.4
numba==0.68.1
numpy==2.5.4
obspy==1.5.1
onnxruntime==1.31.1
openpyxl==3.1.5
packaging==26.3
pandas==3.0.6
pillow==13.0.0
pluggy==1.7.0
propcache==0.5.4
protobuf==7.36.2
psutil==7.2.2
py-spy==0.4.2
pyarrow==26.0.0
pycparser==3.12
Pygments==2.21.0
PyGObject==3.50.0
pyparsing==3.3.3
pytest==9.2.0
python-dateutil==3.0.0
python-docx==1.2.0
python-dotenv==1.2.4
python-pptx==1.0.2
PyWavelets==1.10.0
PyYAML==6.0.3
rank-bm25==0.2.2
redis==6.1.0
regex==2026.9.29
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scipy==1.18.1
sentencepiece==0.2.2
setuptools==78.1.0
shellingham==1.5.4
six==1.17.0
soundfile==0.14.0
soupsieve==3.0.4
SQLAlchemy==2.1.8
sympy==1.14.0
tabulate==0.10.0
tokenizers==0.23.3
torch==2.14.1+cpu
tqdm==4.70.1
transformers==5.22.0
truststore==0.10.5
typer==0.27.3
typing_extensions==4.16.0
tzdata==2026.6
urllib3==2.8.0
websockify==0.12.0
wheel==0.46.1
wrapt==1.15.0
xlsxwriter==3.3.0
xxhash==4.0.2
yarl==1.25.1
yt-dlp==2026.10.21
zstandard==0.25.0
```
