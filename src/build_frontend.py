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


DASHBOARD_HTML = FRONTEND_DIR / "dashboard.html"


def _update_dashboard_html(dashboard_path: Path, stocks: list) -> bool:
    """
    Ensure dashboard.html embeds all stocks so it renders correctly
    even when opened locally under file:/// without CORS/web server.
    """
    if not dashboard_path.exists():
        return False
    
    with open(dashboard_path, "r", encoding="utf-8") as fh:
        html = fh.read()

    stocks_json = json.dumps(stocks)
    
    # Check if EMBEDDED_STOCKS is already defined
    if "const EMBEDDED_STOCKS =" in html:
        html = re.sub(
            r"const EMBEDDED_STOCKS\s*=\s*\[.*?\];",
            f"const EMBEDDED_STOCKS = {stocks_json};",
            html,
            flags=re.DOTALL
        )
    else:
        # Define EMBEDDED_STOCKS right before loadStockTable
        embed_def = f"\n// Pre-embedded 75-stock dataset for direct offline / file:/// navigation\nconst EMBEDDED_STOCKS = {stocks_json};\n\n"
        html = html.replace("function loadStockTable() {", embed_def + "function loadStockTable() {")

    # Update loadStockTable implementation to use renderStockRows and EMBEDDED_STOCKS on catch
    new_load_logic = """function renderStockRows(stocks) {
  const tb = document.getElementById("allocationStockTable");
  if (!tb || !stocks) return;
  tb.innerHTML = stocks.map(s => {
    const name = s.name || s[0];
    const sector = s.sector || s[1] || "Equity";
    const m = (s.m !== undefined ? s.m : s[2]);
    const v = (s.v !== undefined ? s.v : s[3]);
    return `
      <tr>
        <td class="bold">${name}</td>
        <td>${sector}</td>
        <td class="r">${(m * 100).toFixed(1)}%</td>
        <td class="r">${(v * 100).toFixed(1)}%</td>
      </tr>
    `;
  }).join("");
}

function loadStockTable() {
  fetch("results/frontend.json")
    .then(r => r.json())
    .then(data => {
      if (data && data.stocks && data.stocks.length > 0) {
        renderStockRows(data.stocks);
      } else {
        renderStockRows(EMBEDDED_STOCKS);
      }
    })
    .catch(() => {
      // Offline file:/// or CORS fallback: render all point-in-time universe stocks
      renderStockRows(EMBEDDED_STOCKS);
    });
}"""

    pattern = re.compile(r"function loadStockTable\(\)\s*\{.*?\}\s*function drawEquityChart", re.DOTALL)
    if pattern.search(html):
        html = pattern.sub(new_load_logic + "\n\nfunction drawEquityChart", html)

    with open(dashboard_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return True

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
    return f"const S=[{inner}].map(a=>({{n:a[0],s:a[1],m:a[2],v:a[3]}}));"


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
    # 4. Update TEST line from strategy_metrics.csv if available
    # ------------------------------------------------------------------
    strat_csv = RESULTS_DIR / "strategy_metrics.csv"
    if strat_csv.exists():
        try:
            import pandas as pd
            sm_df = pd.read_csv(strat_csv)
            s1_val = round(float(sm_df[sm_df["strategy"] == "S1"]["cagr"].values[0]), 1)
            s4_val = round(float(sm_df[sm_df["strategy"] == "S4"]["cagr"].values[0]), 1)
            s0_val = round(float(sm_df[sm_df["strategy"] == "S0"]["cagr"].values[0]), 1)
            new_test = f"const TEST={{equal:{s1_val},model:{s4_val},nifty:{s0_val}}};"
            new_html = re.sub(r"const TEST=\{[^}]+\};", new_test, new_html)
        except Exception:
            pass

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
    # --- Update dashboard.html if present ---
    if _update_dashboard_html(DASHBOARD_HTML, payload["stocks"]):
        print(f"Updated: {DASHBOARD_HTML} with {len(payload['stocks'])} embedded stocks for offline/file:/// view.")

    print(
        f"Stock count: {len(payload['stocks'])} | "
        f"NIFTY m={payload['nifty']['m']:.3f} v={payload['nifty']['v']:.3f} | "
        f"RHO={payload['rho']:.3f}"
    )


if __name__ == "__main__":
    main()
