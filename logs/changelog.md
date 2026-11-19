# Project Changelog

| Date | Change | Reason | Approver | Stages to Rerun |
| :--- | :--- | :--- | :--- | :--- |
| **2026-11-17** | Baseline specification frozen (SRS v1.0) | Initial Day 1 project setup | Project Guide | None |
| **2026-11-18** | Replaced PyPortfolioOpt with native `scipy.optimize` (SLSQP) & `sklearn` Ledoit-Wolf | Binary compatibility and deterministic convergence under Python 3.12 | All Team Members | Stage 4, Stage 5 |
| **2026-11-18** | Confirmed final universe size at 27 stocks | All 27 stocks have complete 2015-2026 history and clear sector representation | Student 1 & Guide | Stages 1 through 5 |
| **2026-11-19** | S5 holds 20% per stock, above the 15% cap, because 5 equal-weight stocks can't satisfy 15%. SRS §2.4 and §8.6 conflict. | Mathematical necessity: $1/5 = 0.20$, which cannot meet a 15% concentration cap | Guide | Stage 4, Stage 5 |
| **2026-11-19** | Added `filled` boolean flag column in `data/raw/prices.csv` (FR-1.5); removed `.bfill()` | Strict forward-fill only (max 3 days) to eliminate lookahead bias | Guide & Team | Stages 1 through 5 |
| **2026-11-19** | Fixed `src/build_frontend.py` to preserve demo banner; exactly 3 lines changed in `FolioWise_baseline.html` | Conformance with SRS §3.3 permitted edits | Guide & Team | Stage 5 |
