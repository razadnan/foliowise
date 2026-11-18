"""
src/build_frontend.py
=====================
Inject real backtest data from results/frontend.json into the
FolioWise HTML template to produce frontend/index.html.

Usage
-----
    python src/build_frontend.py
    # or from a notebook:
    # import subprocess; subprocess.run(["python", "src/build_frontend.py"], check=True)

What it replaces in the baseline HTML
--------------------------------------
1. ``const S=[...]``          → real stock list from frontend.json
2. ``NIFTY={m:...,v:...}``   → real NIFTY mean/vol from frontend.json
3. ``RHO=...``                → real average pairwise correlation
4. ``"13 large companies"``   → actual stock count
5. The demo banner text       → updated note with stock count
"""

import json
import re
import difflib
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (relative to project root – adjust if running from a different cwd)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_HTML = FRONTEND_DIR / "FolioWise_baseline.html"
OUTPUT_HTML = FRONTEND_DIR / "index.html"
DIFF_FILE = FRONTEND_DIR / "baseline_diff.txt"
FRONTEND_JSON = RESULTS_DIR / "frontend.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stocks_to_js(stocks: list) -> str:
    """
    Convert the stocks list from frontend.json into the JS array literal.

    Each stock: {"ticker": ..., "name": ..., "m": ..., "v": ..., "sector": ...}
    JS format:  [name, sector, m, v]
    """
    items = []
    for s in stocks:
        name = s.get("name", s.get("ticker", "Unknown"))
        sector = s.get("sector", "Equity")
        m = s.get("m", 0.0)
        v = s.get("v", 0.2)
        # Escape quotes in name/sector
        name_esc = name.replace('"', '\\"')
        sector_esc = sector.replace('"', '\\"')
        items.append(f'["{name_esc}","{sector_esc}",{m:.4f},{v:.4f}]')
    inner = ",".join(items)
    return f"const S=[{inner}].map(a=>{{n:a[0],s:a[1],m:a[2],v:a[3]}}));"


def _build_new_html(baseline: str, payload: dict) -> str:
    """
    Apply all substitutions to the baseline HTML and return the new content.
    """
    stocks = payload["stocks"]
    nifty = payload["nifty"]
    rho = payload["rho"]
    n_stocks = len(stocks)

    new_html = baseline

    # ------------------------------------------------------------------
    # 1. Replace const S=[...].map(...)  block
    # ------------------------------------------------------------------
    new_s_line = _stocks_to_js(stocks)
    # Match from 'const S=[' to the closing ';' on the same logical line
    # The original spans one very long line ending with });
    s_pattern = re.compile(
        r"const S=\[.*?\]\.map\(a=>\(\{n:a\[0\],s:a\[1\],m:a\[2\],v:a\[3\]\}\)\);",
        re.DOTALL,
    )
    new_html, count = s_pattern.subn(new_s_line, new_html)
    if count == 0:
        # Fallback: try a looser match
        s_pattern2 = re.compile(r"const S=\[.*?\];", re.DOTALL)
        new_html, count2 = s_pattern2.subn(f"const S={json.dumps([[s['name'], s.get('sector','Equity'), s['m'], s['v']] for s in stocks])}.map(a=>{{n:a[0],s:a[1],m:a[2],v:a[3]}});", new_html)

    # ------------------------------------------------------------------
    # 2. Replace NIFTY={m:...,v:...}
    # ------------------------------------------------------------------
    nifty_replacement = f"NIFTY={{m:{nifty['m']:.4f},v:{nifty['v']:.4f}}}"
    new_html = re.sub(r"NIFTY=\{m:[^}]+\}", nifty_replacement, new_html)

    # ------------------------------------------------------------------
    # 3. Replace RHO=...  (a number)
    # ------------------------------------------------------------------
    new_html = re.sub(r"RHO=[0-9.]+", f"RHO={rho:.4f}", new_html)

    # ------------------------------------------------------------------
    # 4. Replace "13 large companies" with actual count
    # ------------------------------------------------------------------
    new_html = re.sub(
        r"\b13\b large companies",
        f"{n_stocks} large companies",
        new_html,
    )
    # Also fix the demo-banner note if it mentions 13
    new_html = re.sub(
        r"sample figures for 13 large companies",
        f"model data for {n_stocks} large companies",
        new_html,
    )
    # Update demo banner
    new_html = new_html.replace(
        "Demo mode: figures use sample market data, not live prices.",
        f"Live model: figures from ML backtest ({n_stocks} NIFTY 50 stocks, 2015–2026).",
    )

    return new_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load inputs ---
    if not FRONTEND_JSON.exists():
        raise FileNotFoundError(
            f"results/frontend.json not found at {FRONTEND_JSON}.\n"
            "Run notebook 04_backtest.ipynb first."
        )
    if not BASELINE_HTML.exists():
        raise FileNotFoundError(
            f"Baseline HTML not found at {BASELINE_HTML}."
        )

    with open(FRONTEND_JSON, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    with open(BASELINE_HTML, "r", encoding="utf-8") as fh:
        baseline = fh.read()

    # --- Build new HTML ---
    new_html = _build_new_html(baseline, payload)

    # --- Save index.html ---
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
        fh.write(new_html)
    print(f"Saved: {OUTPUT_HTML}")

    # --- Save diff ---
    baseline_lines = baseline.splitlines(keepends=True)
    new_lines = new_html.splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        baseline_lines,
        new_lines,
        fromfile="FolioWise_baseline.html",
        tofile="index.html",
    ))
    with open(DIFF_FILE, "w", encoding="utf-8") as fh:
        fh.writelines(diff)
    n_changed = sum(1 for line in diff if line.startswith("+") or line.startswith("-"))
    print(f"Saved diff ({n_changed} changed lines): {DIFF_FILE}")
    print(
        f"Stock count: {len(payload['stocks'])} | "
        f"NIFTY m={payload['nifty']['m']:.3f} v={payload['nifty']['v']:.3f} | "
        f"RHO={payload['rho']:.3f}"
    )


if __name__ == "__main__":
    main()
