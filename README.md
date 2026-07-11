# Crop Recommendation System — Multi-Version Monorepo

A progression from synthetic toy data → real-world reliable crop recommendation.

## 📦 Versions

| Version | Folder | Data Source | Crops | Real-World Accuracy | Status |
|---------|--------|-------------|-------|---------------------|--------|
| **v0.1** | `v0.1/` | ICFA India (synthetic) | 22 | ~15% | Legacy |
| **v1.0** | `v1.0/` | ICFA India (synthetic) | 22 | ~17% | Baseline GUI + LLM |
| **v2.5** | `v2.5/` | GAEZ + GROW-Africa (synthetic) | 33 | ~17% | Full features |
| **CRS_V2.5_Lite** | `CRS_V2.5_Lite/` | **FUBC/FAO + WorldClim (REAL)** | **29** | **83%** | **Production** |

> **Key finding:** Synthetic datasets (ICFA, GAEZ) have narrow non-overlapping ranges per crop → model memorizes boundaries, fails on real inputs. Real fertilizer survey data (FUBC) has overlapping ranges → model learns patterns, generalizes.

## 🚀 Quick Start (Recommended)

```bash
cd CRS_V2.5_Lite
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**First run** auto-trains the model (~30s).

## 📊 Real-World Performance

| Metric | Synthetic (v2.5) | Real (CRS_V2.5_Lite) |
|--------|------------------|----------------------|
| Exact match (29 agronomy test cases) | 17% | **83%** |
| Top-3 accuracy | 34% | **100%** |
| CV accuracy (29 classes) | 56% | 70% |
| Features | 9 | 7 |

## 🔬 Data Pipeline

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

## 📁 Repository Structure

```
.
├── CRS_V2.5_Lite/           # ← USE THIS (production)
│   ├── main.py              # Clean GUI (~350 lines)
│   ├── xgboost_model.pkl    # Real-data model
│   ├── crop_ranges.json     # 29 crops with real ranges
│   ├── dataset/             # 2,900-row CSV
│   └── requirements.txt
├── v1.0/                    # Legacy (synthetic India data)
├── v2.5/                    # Legacy (synthetic GAEZ data)
├── v0.1/                    # Legacy (original)
└── LICENSE
```

## 📚 Documentation

- **CRS_V2.5_Lite/README.md** — Full academic documentation, reproducibility, citations
- **v1.0/README.md** — Legacy GUI documentation
- **v2.5/scripts/prepare_data.py** — Synthetic data pipeline (for reference)

## 🏷️ Releases

- `v0.1-initial` — First commit
- `v1.0-gui-llm` — CustomTkinter + LLM explanations
- `v2.5-full-features` — Weather, soil profiles, continent encoder
- **`v2.5-lite-real-data`** — **Real FUBC data, 83% real-world accuracy** ← Latest

## 📜 License

MIT — see [LICENSE](LICENSE)

## 📖 Citation

```bibtex
@software{crs_v25_lite,
  title = {CRS_V2.5_Lite: Real-Data Crop Recommendation System},
  author = {Alouhmy, Mohamed},
  year = {2026},
  url = {https://github.com/Exo-del/StupidFarmer-Crop-recommendation-system-with-dumb-chatbot/tree/main/CRS_V2.5_Lite}
}
```

**Data:** FAO/IFA FUBC surveys (Dryad: `10.5061/dryad.2rbnzs7qh`) + WorldClim 2.1
