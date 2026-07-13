# SeedBrain

> AI-powered crop recommendation with real-world accuracy. Built with actual FAO fertilizer data, not synthetic toy sets.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/github/v/release/Exo-del/SeedBrain-CRS?filter=CRS_V2.5_Lite-*&label=latest)](https://github.com/Exo-del/SeedBrain-CRS/releases)
[![Docker](https://img.shields.io/badge/docker-ready-2496ed.svg?logo=docker)](Dockerfile)

A progression from synthetic toy data → real-world reliable crop recommendation.

---

## Versions

| Version | Folder | Data Source | Crops | Real-World Accuracy | Status |
|---------|--------|-------------|-------|---------------------|--------|
| **v0.1** | `v0.1/` | ICFA India (synthetic) | 22 | ~15% | Legacy |
| **v1.0** | `v1.0/` | ICFA India (synthetic) | 22 | ~17% | Baseline GUI + LLM |
| **v2.5** | `v2.5/` | GAEZ + GROW-Africa (synthetic) | 33 | ~17% | Full features |
| **CRS_V2.5_Lite** | `CRS_V2.5_Lite/` | **FUBC/FAO + WorldClim (REAL)** | **29** | **83%** | **Production** |

> **Key finding:** Synthetic datasets (ICFA, GAEZ) have narrow non-overlapping ranges per crop → model memorizes boundaries, fails on real inputs. Real fertilizer survey data (FUBC) has overlapping ranges → model learns patterns, generalizes.

![SeedBrain GUI](CRS_V2.5_Lite/assets/screenshot.png)

---

## Quick Start

### Option 1: Download release (recommended — no clone, pre-trained model)

Grab the latest release archive from the [Releases page](https://github.com/Exo-del/SeedBrain-CRS/releases):

```bash
tar xzf seedbrain.tar.gz
cd CRS_V2.5_Lite
python -m venv venv
source venv/bin/activate
pip install -r requirements.lock
python main.py
```

### Option 2: From source

```bash
cd CRS_V2.5_Lite
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**First run** auto-trains the model (~30s).

### Option 3: Docker

```bash
docker build -t seedbrain .
# Run with X11 forwarding (Linux):
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix seedbrain
```

---

## Real-World Performance

| Metric | Synthetic (v2.5) | Real (CRS_V2.5_Lite) |
|--------|------------------|----------------------|
| Exact match (29 agronomy test cases) | 17% | **83%** |
| Top-3 accuracy | 34% | **100%** |
| CV accuracy (29 classes) | 56% | 70% |
| Features | 9 | 7 |

---

## Data Pipeline

```
FUBC (FAO/IFA) Fertilizer Use By Crop 1978–2019
    │
    ├─► 111 countries × 516 crops × N/P/K rates (kg/ha)
    │
    ├─► WorldClim 2.1 → country climate normals (temp/humidity/rainfall)
    │
    ├─► Crop-specific pH from FAO/USDA guides
    │
    └─► 2,900 samples × 29 crops × 7 features (balanced 100 each)
```

---

## Repository Structure

```
.
├── .github/workflows/       # CI: automated release builder
├── CRS_V2.5_Lite/           # ← USE THIS (production)
│   ├── main.py              # Clean GUI (~350 lines)
│   ├── xgboost_model.pkl    # Real-data model (auto-trained)
│   ├── crop_ranges.json     # 29 crops with real ranges
│   ├── dataset/             # 2,900-row CSV
│   ├── requirements.txt     # Loose deps
│   ├── requirements.lock    # Pinned deps (reproducible)
│   └── README.md            # Full academic docs
├── Dockerfile               # Containerized reproducible run
├── release.sh               # Manual release builder script
├── v1.0/                    # Legacy (synthetic India data)
├── v2.5/                    # Legacy (synthetic GAEZ data)
├── v0.1/                    # Legacy (original)
└── LICENSE
```

---

## Documentation

- **CRS_V2.5_Lite/README.md** — Full academic documentation, reproducibility, citations
- **v1.0/README.md** — Legacy GUI documentation
- **v2.5/scripts/prepare_data.py** — Synthetic data pipeline (for reference)

---

## Releases

Each release ships **pre-built model artifacts** + **pinned dependencies** — no clone, no training.

| Tag | Description | Assets |
|-----|-------------|--------|
| `v0.1-initial` | First commit | Source only |
| `v1.0-gui-llm` | CustomTkinter + LLM explanations | Source only |
| `v2.5-full-features` | Weather, soil profiles, continent encoder | Source only |
| `CRS_V2.5_Lite-*` | **Real FUBC data, 83% accuracy** | `seedbrain.tar.gz`, `seedbrain.zip`, `checksums.txt` |

New releases are **automated** — push a tag matching `v*` or `CRS_V2.5_Lite-*`:
```bash
# Manual release script (run from repo root):
./release.sh CRS_V2.5_Lite-20260713

# Or push a tag (triggers GitHub Actions):
git tag CRS_V2.5_Lite-20260713
git push origin CRS_V2.5_Lite-20260713
```

Release artifacts include: `main.py`, pre-trained `xgboost_model.pkl`, `label_encoder.pkl`, `crop_ranges.json`, `dataset/`, `requirements.lock` (pinned deps), and checksums.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@software{seedbrain,
  title = {SeedBrain: Real-Data Crop Recommendation System},
  author = {Alouhmy, Mohamed},
  affiliation = {Faculté des Sciences Ain Chock (FSAC), Université Hassan II de Casablanca},
  year = {2026},
  url = {https://github.com/Exo-del/SeedBrain-CRS}
}
```

**Data:** FAO/IFA FUBC surveys (Dryad: `10.5061/dryad.2rbnzs7qh`) + WorldClim 2.1

---

## Links

- **Releases:** https://github.com/Exo-del/SeedBrain-CRS/releases
- **Issues:** https://github.com/Exo-del/SeedBrain-CRS/issues

---

**Built with real data, for real farmers.** 🌱🧠
