"""Step 4 - Build the interactive dashboard (docs/index.html, served by GitHub Pages).

Inputs  (outputs/): dashboard_data.json
Outputs (docs/):    index.html - one self-contained page; charts use Plotly from its CDN.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src" / "dashboard_template.html"

if __name__ == "__main__":
    data = json.loads((ROOT / "outputs" / "dashboard_data.json").read_text(encoding="utf-8"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote docs/index.html ({len(html) / 1024:.0f} KB)")
