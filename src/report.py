"""Refresh the results tables in docs/REPORT.md from the saved results.json files."""
import json
import math
import re
from pathlib import Path

from src.config import Config

REPORT_PATH = Path(__file__).resolve().parent.parent / "docs" / "REPORT.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _fmt(v: float, nd: int) -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def _volatility_table(r: dict) -> str:
    rows = ["### Volatility (log units)", "",
            "| | RMSE | MAE | R² | Correlation |", "|---|---|---|---|---|"]
    for name, m in r["metrics"].items():
        label = f"**{name}**" if name == "MODEL" else name
        cells = [_fmt(m["rmse"], 3), _fmt(m["mae"], 3), _fmt(m["r2_vs_train_mean"], 3), _fmt(m["corr"], 3)]
        rows.append(f"| {label} | " + " | ".join(f"**{c}**" if name == "MODEL" else c for c in cells) + " |")
    return "\n".join(rows)


def _return_table(r: dict) -> str:
    rows = ["### Return", "",
            "| | Return RMSE | Return MAE | Direction correct % |", "|---|---|---|---|"]
    for name, m in {"MODEL": r["model"], **r["baselines"]}.items():
        label = f"**{name}**" if name == "MODEL" else name
        rows.append(f"| {label} | {_fmt(m['return_rmse'], 6)} | {_fmt(m['return_mae'], 6)} | "
                    f"{_fmt(m['directional_accuracy_%'], 1)} |")
    d = r["direction_test"]
    rows += ["", f"Direction right on {d['hits']} of {d['n']} days. Always \"up\" gives "
                 f"{d['majority_rate_%']:.1f}%. Binomial test p = {d['p_value']:.2f}."]
    return "\n".join(rows)


def update_report(ticker: str, output_dir: Path = Config.output_dir, report_path: Path = REPORT_PATH) -> bool:
    """Rewrite the block between the RESULTS markers with tables built from the ticker's latest results.json files.

    Returns True if the report was updated. Missing files or markers are skipped with a message, never an error.
    """
    name = ticker.replace("^", "")
    sections = []
    for suffix, build in (("_lstm_vol", _volatility_table), ("_lstm", _return_table)):
        path = Path(output_dir) / f"{name}{suffix}" / "results.json"
        if path.exists():
            sections.append(build(json.loads(path.read_text())))
    if not sections:
        print(f"Report not updated: no results found for {ticker}.")
        return False
    if not report_path.exists():
        print(f"Report not updated: {report_path} does not exist.")
        return False
    text = report_path.read_text(encoding="utf-8")
    if START not in text or END not in text:
        print(f"Report not updated: markers {START} / {END} not found in {report_path}.")
        return False
    header = f"Source: `outputs/{name}_lstm_vol/results.json` and `outputs/{name}_lstm/results.json` (auto-generated)."
    block = f"{START}\n{header}\n\n" + "\n\n".join(sections) + f"\n{END}"
    new = re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _: block, text, flags=re.S)
    report_path.write_text(new, encoding="utf-8")
    print(f"Updated {report_path}")
    return True
