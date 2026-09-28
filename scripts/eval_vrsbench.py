"""
scripts/eval_vrsbench.py
Evaluate the SatQuery AI pipeline on a local VRSBench test subset.
Writes results/vrsbench.json with sample count, metric, and date.

Usage:
    python scripts/eval_vrsbench.py --data /path/to/VRSBench_test
"""
from __future__ import annotations
import argparse
import json
import datetime
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def accuracy(predictions: list, references: list) -> float:
    if not predictions:
        return 0.0
    correct = sum(p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references))
    return correct / len(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to VRSBench test directory")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset path not found: {data_path}")
        print("Download VRSBench from the official release and point --data to the test split.")
        return 1

    # Try to load VQA annotations
    qa_file = data_path / "qa_pairs.json"
    if not qa_file.exists():
        qa_file = next(data_path.glob("*qa*.json"), None)
    if not qa_file:
        print("ERROR: Could not find qa_pairs.json in the dataset directory.")
        return 1

    qa_pairs = json.loads(qa_file.read_text())
    if not qa_pairs:
        print("ERROR: No QA pairs found.")
        return 1

    print(f"Loaded {len(qa_pairs)} QA pairs from VRSBench.")
    print("Running pipeline inference…")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

    from app.ingestion import load_image
    from app.tools.vqa import run_vqa, measure_scene
    from app.evidence import EvidenceBundle

    preds, refs = [], []
    n_failed = 0

    for i, item in enumerate(qa_pairs[:500]):  # cap at 500 samples for speed
        img_path = data_path / item.get("image", "")
        question = item.get("question", "")
        ref = item.get("answer", "")

        if not img_path.exists():
            n_failed += 1
            continue

        try:
            data, meta = load_image(img_path)
            bundle = EvidenceBundle(f"eval_{i}")
            result = run_vqa(data, meta, question, bundle, {})
            preds.append(result.answer)
            refs.append(ref)
        except Exception as e:
            preds.append("")
            refs.append(ref)
            n_failed += 1

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(qa_pairs)} processed…")

    acc = accuracy(preds, refs)
    result_data = {
        "dataset": "VRSBench",
        "task": "VQA",
        "sample_count": len(preds),
        "failed": n_failed,
        "metric": "exact_match_accuracy",
        "score": round(acc, 4),
        "date": datetime.datetime.utcnow().isoformat() + "Z",
        "note": "Exact-match accuracy; case-insensitive strip comparison.",
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "vrsbench.json"
    out.write_text(json.dumps(result_data, indent=2))
    print(f"\nResult: {acc:.4f} exact-match accuracy ({len(preds)} samples)")
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
