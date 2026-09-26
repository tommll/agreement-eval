"""Self-contained HTML export of a run.

One file, no network: figures are embedded as data URIs and the CSS is inline,
so the report opens from disk, survives being emailed, and prints. It renders
from the same AnalysisResult as the markdown report, reusing its section order
and its verdict text - the two must never disagree about what a run found.
"""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .analysis import AnalysisResult
from .report import SECTIONS, TABLE_NOTES, _fmt, _verdict, _yaml_ish

# Figures, in reading order, with the question each one answers.
# Each caption says what to look for - the chart already carries its own title.
FIGURES: Sequence[Tuple[str, str]] = (
    ("agreement_vs_accuracy.png",
     "If agreement predicted correctness, these bars would climb left to right, and the climb "
     "would survive inside the populated-gold series."),
    ("calibration.png",
     "Distance from the dashed line is how wrong it is to read an agreement rate as an accuracy."),
    ("aggregators.png",
     "Aggregated methods (blue) against each config alone (orange), on identical items. The "
     "oracle bar is the ceiling any aggregator could reach."),
    ("per_field_trust.png",
     "Where the blue bar fails to clear the orange one, unanimity on that field buys nothing "
     "over the base rate."),
)

STYLE = """
:root {
  color-scheme: light dark;
  --surface: #fcfcfb; --panel: #ffffff; --ink: #14140f; --ink-2: #52514e;
  --muted: #898781; --rule: #e1e0d9; --accent: #2a78d6; --warn-bg: #fdf6ec;
  --warn-ink: #7a4a10; --warn-rule: #eda100;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --panel: #212120; --ink: #f4f4f0; --ink-2: #c3c2b7;
    --muted: #8d8b84; --rule: #34342f; --accent: #6da7ec; --warn-bg: #2a2213;
    --warn-ink: #edc98a; --warn-rule: #a97c12;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface); color: var(--ink);
  font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 40px 24px 80px; }
header { border-bottom: 1px solid var(--rule); padding-bottom: 20px; margin-bottom: 32px; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 19px; margin: 44px 0 14px; letter-spacing: -0.01em; }
h3 { font-size: 14px; margin: 26px 0 8px; color: var(--ink-2); font-weight: 600;
     font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
p { margin: 0 0 12px; } strong { font-weight: 650; }
.sub { color: var(--muted); font-size: 13px; }
.note { color: var(--ink-2); font-size: 13.5px; margin: 0 0 10px; }
.callout {
  background: var(--warn-bg); color: var(--warn-ink); border-left: 3px solid var(--warn-rule);
  padding: 12px 16px; margin: 16px 0; border-radius: 0 6px 6px 0; font-size: 14px;
}
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0 8px; }
.tile { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px; }
.tile .k { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; }
.tile .v { font-size: 25px; font-weight: 600; margin-top: 4px; font-variant-numeric: tabular-nums; }
.tile .v.small { font-size: 18px; }
.scroll { overflow-x: auto; margin: 0 0 18px; border: 1px solid var(--rule); border-radius: 8px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--rule); white-space: nowrap; }
th { background: var(--surface); color: var(--ink-2); font-weight: 600; position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: color-mix(in srgb, var(--accent) 7%, transparent); }
figure { margin: 0 0 26px; }
figure img { width: 100%; height: auto; display: block; border: 1px solid var(--rule); border-radius: 8px; background: #fcfcfb; }
figcaption { color: var(--muted); font-size: 12.5px; margin-top: 7px; }
details { margin-top: 28px; border-top: 1px solid var(--rule); padding-top: 14px; }
summary { cursor: pointer; color: var(--ink-2); font-size: 14px; }
pre { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px;
      padding: 14px; overflow-x: auto; font-size: 12.5px; line-height: 1.5; }
ul { margin: 0 0 12px; padding-left: 20px; } li { margin-bottom: 6px; }
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--rule); color: var(--muted); font-size: 12.5px; }
@media print { body { background: #fff; } .scroll { overflow: visible; } }
"""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=False)


def _inline_markup(text: str) -> str:
    """The verdict text carries **bold** and `code`; keep it, escape the rest."""
    out = _esc(text)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    return out


def _table(frame: pd.DataFrame, index: bool = False, max_rows: int = 200) -> str:
    if frame is None or frame.empty:
        return '<p class="note">(no rows)</p>'
    df = frame.head(max_rows)
    if index:
        df = df.reset_index()
    head = "".join(f"<th>{_esc(c)}</th>" for c in df.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_esc(_fmt(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    more = (f'<p class="note">{len(frame) - max_rows} more rows in the CSV.</p>'
            if len(frame) > max_rows else "")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>{more}'


def _tiles(headline: Dict[str, Any]) -> str:
    populated = headline.get("populated") or {}
    aggregation = headline.get("aggregation") or {}
    cells: List[Tuple[str, str, bool]] = [
        ("items", _fmt(headline.get("n_items")), False),
        ("unanimity rate", _fmt(headline.get("unanimity_rate")), False),
        ("correct if unanimous", _fmt(headline.get("p_correct_given_unanimous")), False),
        ("correct if split", _fmt(headline.get("p_correct_given_split")), False),
        ("agreement AUC", _fmt(headline.get("auc_pairwise_agreement")), False),
        ("calibration error", _fmt(headline.get("ece_agreement_vs_accuracy")), False),
    ]
    if aggregation:
        cells.append(("Dawid-Skene vs majority", _fmt(aggregation.get("ds_minus_mv")), False))
        cells.append(("ceiling: any config", _fmt(aggregation.get("oracle_any_config")), False))
    if populated.get("n"):
        cells.append(("populated items", _fmt(populated.get("n")), False))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{_esc(k)}</div>'
        f'<div class="v{" small" if small else ""}">{_esc(v)}</div></div>'
        for k, v, small in cells
    ) + "</div>"


def _figures(figures_dir: Path) -> str:
    blocks: List[str] = []
    for filename, caption in FIGURES:
        path = figures_dir / filename
        if not path.exists():
            continue
        data = base64.b64encode(path.read_bytes()).decode()
        blocks.append(
            f'<figure><img alt="{_esc(caption)}" src="data:image/png;base64,{data}">'
            f"<figcaption>{_esc(caption)}</figcaption></figure>"
        )
    return f"<h2>Figures</h2>{''.join(blocks)}" if blocks else ""


def _judge_section(headline: Dict[str, Any]) -> str:
    check = headline.get("judge_check")
    if not check:
        return ""
    by_kind = check.get("by_kind") or {}
    frame = pd.DataFrame(
        [{"control": k, "n": v.get("n"), "judge accuracy": v.get("accuracy")} for k, v in sorted(by_kind.items())]
    )
    identical = (by_kind.get("identical") or {}).get("accuracy")
    false_eq = check.get("false_equivalence_rate")
    warn = ""
    if (isinstance(identical, float) and identical < 0.95) or (isinstance(false_eq, float) and false_eq > 0.1):
        warn = ('<div class="callout">This judge does not pass its own controls, so its verdicts are '
                "not used in the numbers below. A judge that cannot recognise two identical strings - "
                "or that merges unrelated values - would manufacture exactly the agreement this study "
                "is trying to measure.</div>")
    return (
        "<h2>Judge validation</h2>"
        f'<p class="note">Controls with known answers, judged by '
        f"{_esc(check.get('provider'))}/{_esc(check.get('model'))}: identical strings, cosmetic "
        f"rewrites, and values taken from different documents.</p>"
        f"{_table(frame)}"
        f"<p>Overall {_esc(_fmt(check.get('accuracy')))}; false-equivalence rate on values that are "
        f"genuinely different: {_esc(_fmt(false_eq))}.</p>{warn}"
    )


def render_html(
    result: AnalysisResult,
    experiment: Optional[Dict[str, Any]] = None,
    figures_dir: Optional[Path] = None,
) -> str:
    headline = result.headline
    n_configs = len(result.tables.get("accuracy_per_config", pd.DataFrame()))

    # Markdown blockquotes in the verdict are the caveats; they become callouts.
    verdict_html = "".join(
        f'<div class="callout">{_inline_markup(block.lstrip("> ").strip())}</div>'
        if block.lstrip().startswith(">") else f"<p>{_inline_markup(block.strip())}</p>"
        for block in _verdict(headline).split("\n") if block.strip()
    )

    body: List[str] = []
    for title, names in SECTIONS:
        available = [n for n in names
                     if isinstance(result.tables.get(n), pd.DataFrame) and not result.tables[n].empty]
        if not available:
            continue
        body.append(f"<h2>{_esc(title.replace('*', ''))}</h2>")
        for name in available:
            body.append(f"<h3>{_esc(name)}</h3>")
            if name in TABLE_NOTES:
                body.append(f'<p class="note">{_esc(TABLE_NOTES[name])}</p>')
            body.append(_table(result.tables[name], index=name.endswith("_matrix")))

    config_block = (
        f"<details><summary>Experiment configuration</summary><pre>{_esc(_yaml_ish(experiment))}</pre></details>"
        if experiment else ""
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agreement vs. accuracy - {_esc(result.run_id)}</title>
<style>{STYLE}</style></head>
<body><div class="wrap">
<header>
  <h1>Does model agreement predict accuracy?</h1>
  <p class="sub">Run <code>{_esc(result.run_id)}</code> &middot; schema <code>{_esc(result.schema_name)}</code>
     &middot; {n_configs} configs &middot; {_esc(_fmt(headline.get('n_items')))} items (document x field)</p>
</header>
<h2>Verdict</h2>
{_tiles(headline)}
{verdict_html}
{_judge_section(headline)}
{_figures(figures_dir or Path("."))}
{"".join(body)}
<h2>Method notes</h2>
<ul>
  <li>Agreement and accuracy use the <em>same</em> normalizer per field type, so neither is measured
      more leniently than the other.</li>
  <li><code>null</code> means normalized-absent: "N/A", "-" and "" all count as null, on both sides.</li>
  <li>Items are (document, field) pairs. An item only one config answered is excluded: with a single
      rater there is nothing to agree about.</li>
  <li>Dawid-Skene over open-vocabulary values uses the one-coin model; the populated/null decision
      additionally gets a full 2x2 confusion-matrix model.</li>
</ul>
{config_block}
<footer>Generated by agreement-eval from run <code>{_esc(result.run_id)}</code>.
Every table on this page is also a CSV next to this file.</footer>
</div></body></html>"""


def write_html(
    result: AnalysisResult,
    outdir: Path,
    experiment: Optional[Dict[str, Any]] = None,
) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "report.html"
    path.write_text(render_html(result, experiment, figures_dir=outdir), encoding="utf-8")
    return path


def open_in_browser(path: Path) -> bool:
    """Open the file in the default browser; False if the environment has none."""
    import webbrowser

    try:
        return webbrowser.open(Path(path).resolve().as_uri())
    except Exception:
        return False
