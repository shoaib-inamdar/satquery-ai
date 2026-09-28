"""
scripts/eval_rsvqa.py
Evaluate on a local RSVQA test subset.
Usage:
    python scripts/eval_rsvqa.py --data /path/to/RSVQA_test
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
        print(f"ERROR: {data_path} not found. Download RSVQA from https://rsvqa.sylvainlobry.com/")
        return 1

    qa_file = next(data_path.glob("*.json"), None)
    if not qa_file:
        print("ERROR: No JSON files found in dataset directory.")
        return 1

    qa_pairs = json.loads(qa_file.read_text())
    if isinstance(qa_pairs, dict):
        qa_pairs = qa_pairs.get("questions", []) or qa_pairs.get("qa", [])

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
    from app.ingestion import load_image
    from app.tools.vqa import run_vqa
    from app.evidence import EvidenceBundle

    preds, refs = [], []
    for i, item in enumerate(qa_pairs[:500]):
        img_path = data_path / item.get("img_id", item.get("image", ""))
        if not Path(img_path).exists():
            continue
        try:
            data, meta = load_image(Path(img_path))
            bundle = EvidenceBundle(f"rsvqa_{i}")
            r = run_vqa(data, meta, item.get("question", ""), bundle, {})
            preds.append(r.answer.strip().lower())
            refs.append(str(item.get("answer", "")).strip().lower())
        except Exception:
            pass

    acc = sum(p == r for p, r in zip(preds, refs)) / max(len(preds), 1)
    result_data = {
        "dataset": "RSVQA",
        "task": "VQA",
        "sample_count": len(preds),
        "metric": "exact_match_accuracy",
        "score": round(acc, 4),
        "date": datetime.datetime.utcnow().isoformat() + "Z",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "rsvqa.json"
    out.write_text(json.dumps(result_data, indent=2))
    print(f"RSVQA accuracy: {acc:.4f} ({len(preds)} samples) → {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
