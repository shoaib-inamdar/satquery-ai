"""
scripts/eval_cdvqa.py
Evaluate on a local CDVQA test subset (change detection VQA).
Usage:
    python scripts/eval_cdvqa.py --data /path/to/CDVQA_test
"""
from __future__ import annotations
import argparse, json, datetime, sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Download CDVQA from the paper's repository.")
        return 1

    qa_file = next(data_path.glob("*.json"), None)
    if not qa_file:
        print("ERROR: No JSON files found in dataset directory.")
        return 1

    qa_pairs = json.loads(qa_file.read_text())
    if isinstance(qa_pairs, dict):
        qa_pairs = list(qa_pairs.values())[0] if qa_pairs else []

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.ingestion import load_image
    from app.tools.change import run_change
    from app.evidence import EvidenceBundle

    preds, refs = [], []
    for i, item in enumerate(qa_pairs[:200]):
        img1 = data_path / item.get("img1", "")
        img2 = data_path / item.get("img2", "")
        if not (Path(img1).exists() and Path(img2).exists()):
            continue
        try:
            d1, m1 = load_image(Path(img1))
            d2, m2 = load_image(Path(img2))
            bundle = EvidenceBundle(f"cdvqa_{i}")
            r = run_change(d1, m1, d2, m2, item.get("question", ""), bundle, {})
            preds.append(r.answer.strip().lower())
            refs.append(str(item.get("answer", "")).strip().lower())
        except Exception:
            pass

    acc = sum(p == r for p, r in zip(preds, refs)) / max(len(preds), 1)
    result_data = {
        "dataset": "CDVQA",
        "task": "Change-VQA",
        "sample_count": len(preds),
        "metric": "exact_match_accuracy",
        "score": round(acc, 4),
        "date": datetime.datetime.utcnow().isoformat() + "Z",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "cdvqa.json"
    out.write_text(json.dumps(result_data, indent=2))
    print(f"CDVQA accuracy: {acc:.4f} ({len(preds)} samples) → {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
