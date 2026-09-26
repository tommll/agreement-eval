"""Command line interface.

Typical flow:

    agreement-eval initdb
    agreement-eval ingest --dataset sroie --split test --limit 200
    agreement-eval extract  --experiment experiments/sroie.yaml
    agreement-eval judge    --experiment experiments/sroie.yaml
    agreement-eval evaluate --run-id sroie-v1

or `agreement-eval all --experiment experiments/sroie.yaml` for all four, and
`agreement-eval demo` for an offline end-to-end run with the mock provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import db
from .analysis import analyze
from .config import Experiment, load_experiment
from .ingest import load_dataset
from .normalize import DEFAULT_JACCARD_THRESHOLD
from .report import write_report

DEFAULT_REPORT_DIR = Path("reports")


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_initdb(args: argparse.Namespace) -> int:
    with db.connect(args.dsn) as conn:
        db.init_db(conn)
    print(f"schema applied to {args.dsn or db.dsn()}")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    docs = load_dataset(
        args.dataset,
        split=args.split,
        limit=args.limit,
        source=args.source,
        download_images=args.download_images,
        root=args.root,
        seed=args.seed,
    )
    with db.connect(args.dsn) as conn:
        db.init_db(conn)
        n = db.upsert_documents(conn, docs)
    populated = sum(1 for d in docs for v in d.gold.values() if v)
    total_fields = sum(len(d.gold) for d in docs)
    print(f"ingested {n} documents from {args.dataset}"
          f" ({populated}/{total_fields} gold fields populated, "
          f"{total_fields - populated} null)")
    return 0


def _documents_for(conn, experiment: Experiment, limit: Optional[int]) -> List[Any]:
    docs = db.fetch_documents(conn, experiment.dataset, experiment.split, limit or experiment.limit)
    if not docs:
        raise SystemExit(
            f"no documents for dataset={experiment.dataset!r} split={experiment.split!r} - run `ingest` first"
        )
    return docs


def cmd_extract(args: argparse.Namespace) -> int:
    from .extract.runner import run_extraction

    experiment = load_experiment(args.experiment)
    if args.run_id:
        experiment.run_id = args.run_id
    with db.connect(args.dsn) as conn:
        db.init_db(conn)
        docs = _documents_for(conn, experiment, args.limit)
        stats = run_extraction(conn, experiment, docs, resume=not args.no_resume,
                               show_progress=not args.quiet)
    _print_json({"run_id": experiment.run_id, "documents": len(docs), **stats})
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    from .judge import check_judge, run_judging
    from .metrics import accuracy
    from .schemas import get_schema

    experiment = load_experiment(args.experiment) if args.experiment else None
    run_id = args.run_id or (experiment.run_id if experiment else None)
    if not run_id:
        raise SystemExit("judge needs --experiment or --run-id")
    with db.connect(args.dsn) as conn:
        info = db.run_info(conn, run_id)
        if info is None:
            raise SystemExit(f"run {run_id!r} not found")
        schema = get_schema(info["schema_name"])
        raw = db.load_predictions(conn, run_id)
        if raw.empty:
            raise SystemExit(f"run {run_id!r} has no extractions yet")
        preds = accuracy.build_predictions(raw, schema)
        cfg = experiment.judge if experiment else None
        if cfg is None:
            from .config import JudgeConfig
            cfg = JudgeConfig(model=args.model)
        if args.check:
            # Validate the judge before trusting it to merge values.
            summary = check_judge(cfg, preds, n_per_kind=args.check_size, show_progress=not args.quiet)
            db.save_analysis(conn, run_id, "judge_check", summary)
            conn.commit()
            _print_json({k: v for k, v in summary.items() if k != "examples"})
            return 0
        if not cfg.enabled:
            print("judge disabled in experiment config; nothing to do")
            return 0
        stats = run_judging(conn, run_id, preds, cfg, show_progress=not args.quiet)
    _print_json({"run_id": run_id, **stats})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment) if args.experiment else None
    run_id = args.run_id or (experiment.run_id if experiment else None)
    if not run_id:
        raise SystemExit("evaluate needs --run-id or --experiment")
    outdir = Path(args.outdir) if args.outdir else DEFAULT_REPORT_DIR / run_id

    with db.connect(args.dsn) as conn:
        result = analyze(conn, run_id, jaccard_threshold=getattr(args, "jaccard_threshold", None)
                         or DEFAULT_JACCARD_THRESHOLD)
    result.write_csvs(outdir)
    report_path = write_report(result, outdir, experiment.to_dict() if experiment else None)

    if not args.no_plots:
        try:
            from .plots import write_plots
            write_plots(result, outdir)
        except Exception as exc:  # plots are a nicety, never a blocker
            print(f"warning: plots skipped ({type(exc).__name__}: {exc})", file=sys.stderr)

    html_path = None
    if args.html or args.open:
        # Written after the plots so the figures can be embedded in it.
        from .html_report import open_in_browser, write_html
        html_path = write_html(result, outdir, experiment.to_dict() if experiment else None)
        if args.open and not open_in_browser(html_path):
            print(f"no browser available; open {html_path} manually", file=sys.stderr)

    if not args.quiet:
        _print_json(result.headline)
    print(f"\nreport: {report_path}\ntables: {outdir}/*.csv"
          + (f"\nhtml:   {html_path}" if html_path else ""))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    experiment = load_experiment(args.experiment)
    ingest_args = argparse.Namespace(
        dsn=args.dsn, dataset=experiment.dataset, split=experiment.split,
        limit=experiment.limit, source=experiment.ingest.get("source", "api"),
        download_images=experiment.ingest.get("download_images", False),
        root=experiment.ingest.get("root"), seed=experiment.ingest.get("seed", 7),
    )
    if not args.skip_ingest:
        cmd_ingest(ingest_args)
    cmd_extract(argparse.Namespace(dsn=args.dsn, experiment=args.experiment, run_id=None,
                                   limit=None, no_resume=False, quiet=args.quiet))
    if experiment.judge.enabled and not args.skip_judge:
        cmd_judge(argparse.Namespace(dsn=args.dsn, experiment=args.experiment, run_id=None,
                                     model=experiment.judge.model, quiet=args.quiet,
                                     check=False, check_size=6))
    return cmd_evaluate(argparse.Namespace(dsn=args.dsn, experiment=args.experiment, run_id=None,
                                           outdir=args.outdir, no_plots=args.no_plots,
                                           html=getattr(args, "html", False),
                                           open=getattr(args, "open", False), quiet=False))


def cmd_demo(args: argparse.Namespace) -> int:
    """Offline end-to-end run: synthetic receipts + mock configs, no API key."""
    experiment_path = args.experiment or str(Path(__file__).resolve().parents[2] / "experiments" / "demo.yaml")
    return cmd_all(argparse.Namespace(
        dsn=args.dsn, experiment=experiment_path, outdir=args.outdir,
        no_plots=args.no_plots, quiet=args.quiet, skip_ingest=False, skip_judge=True,
        html=getattr(args, "html", False), open=getattr(args, "open", False),
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agreement-eval", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: $DATABASE_URL)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("initdb", help="create tables")
    p.set_defaults(func=cmd_initdb)

    p = sub.add_parser("ingest", help="load a dataset into Postgres")
    p.add_argument("--dataset", required=True, choices=["sroie", "funsd", "synthetic"])
    p.add_argument("--split", default=None, help="dataset split (sroie: train|test)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--source", default="api", choices=["api", "parquet"],
                   help="sroie: HF datasets-server rows API, or the full parquet download")
    p.add_argument("--download-images", action="store_true", help="fetch images for vision configs")
    p.add_argument("--root", default=None, help="funsd: path to the extracted dataset")
    p.add_argument("--seed", type=int, default=7, help="synthetic: generator seed")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("extract", help="run every config over every document")
    p.add_argument("--experiment", required=True)
    p.add_argument("--run-id", default=None, help="override the run id in the experiment file")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="re-run calls that already succeeded")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("judge", help="LLM-judge disagreeing pairs and wrong-looking predictions")
    p.add_argument("--experiment", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--model", default="claude-opus-5")
    p.add_argument("--check", action="store_true",
                   help="validate the judge on controls with known answers instead of judging")
    p.add_argument("--check-size", type=int, default=6, help="controls per kind for --check")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("evaluate", help="compute every metric and write the report")
    p.add_argument("--run-id", default=None)
    p.add_argument("--experiment", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--html", action="store_true",
                   help="also write a self-contained report.html (figures embedded)")
    p.add_argument("--open", action="store_true",
                   help="write the HTML report and open it in the browser")
    p.add_argument("--quiet", action="store_true", help="skip the headline JSON dump")
    p.add_argument("--jaccard-threshold", type=float, default=DEFAULT_JACCARD_THRESHOLD,
                   help="token-set Jaccard to the label at which a free-text field counts as correct "
                        f"(default {DEFAULT_JACCARD_THRESHOLD})")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("all", help="ingest + extract + judge + evaluate")
    p.add_argument("--experiment", required=True)
    p.add_argument("--outdir", default=None)
    p.add_argument("--html", action="store_true", help="also write report.html")
    p.add_argument("--open", action="store_true", help="write report.html and open it")
    p.add_argument("--skip-ingest", action="store_true")
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_all)

    p = sub.add_parser("demo", help="offline end-to-end run (mock provider, no API key)")
    p.add_argument("--experiment", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--html", action="store_true", help="also write report.html")
    p.add_argument("--open", action="store_true", help="write report.html and open it")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
