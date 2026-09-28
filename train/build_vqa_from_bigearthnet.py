"""
train/build_vqa_from_bigearthnet.py
Converts BigEarthNet(-MM) labels into templated VQA question-answer pairs.
Follows the RSVQA+BigEarthNet paper approach.

Usage:
    python train/build_vqa_from_bigearthnet.py \\
        --bigearthnet /path/to/BigEarthNet-MM \\
        --subset 5000 \\
        --out train/vqa_pairs.json
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

QUESTION_TEMPLATES = [
    # Presence
    ("Is there {cls} in this image?",
     lambda cls, labs: "yes" if cls in labs else "no"),
    # Dominant class
    ("What is the dominant land cover in this image?",
     lambda cls, labs: labs[0] if labs else "unknown"),
    # Count classes
    ("How many land-cover classes are present in this image?",
     lambda cls, labs: str(len(labs))),
    # Yes/no for multiple classes
    ("Does this image contain both {cls} and {cls2}?",
     None),  # handled specially
]

COMMON_CLASSES = [
    "Continuous urban fabric",
    "Discontinuous urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Water bodies",
    "Coastal lagoons",
    "Estuaries",
    "Sea and ocean",
]


def load_bigearthnet_labels(be_root: Path, subset: int) -> list[dict]:
    """
    Try to load BigEarthNet labels from JSON files under be_root.
    Returns list of {'patch_id': ..., 'labels': [...]} dicts.
    Falls back to a note if the dataset is not present.
    """
    patches = []
    # BigEarthNet-MM stores labels in <patch>/labels_metadata.json
    label_files = sorted(be_root.glob("*/labels_metadata.json"))[:subset]
    if not label_files:
        # Try CSV-style (BigEarthNet v1.0)
        label_files = sorted(be_root.glob("*_labels_metadata.json"))[:subset]
    if not label_files:
        return []  # Dataset not present

    for lf in label_files:
        try:
            data = json.loads(lf.read_text())
            patch_id = lf.parent.name if lf.parent.name != be_root.name else lf.stem
            labels = data.get("labels", data.get("label", []))
            if isinstance(labels, str):
                labels = [labels]
            patches.append({"patch_id": patch_id, "labels": labels})
        except Exception:
            pass
    return patches


def build_pairs(patches: list[dict], rng: random.Random) -> list[dict]:
    pairs = []
    for p in patches:
        labs = p["labels"]
        patch_id = p["patch_id"]
        if not labs:
            continue
        # Presence questions (3 per patch)
        sample_cls = rng.sample(COMMON_CLASSES, min(3, len(COMMON_CLASSES)))
        for cls in sample_cls:
            answer = "yes" if cls in labs else "no"
            pairs.append({
                "patch_id": patch_id,
                "question": f"Is there {cls.lower()} in this image?",
                "answer": answer,
                "type": "presence",
            })
        # Dominant class
        pairs.append({
            "patch_id": patch_id,
            "question": "What is the dominant land cover in this image?",
            "answer": labs[0] if labs else "unknown",
            "type": "dominant_class",
        })
        # Count
        pairs.append({
            "patch_id": patch_id,
            "question": "How many land-cover classes are present in this image?",
            "answer": str(len(labs)),
            "type": "count",
        })
        # Two-class presence
        if len(COMMON_CLASSES) >= 2:
            c1, c2 = rng.sample(COMMON_CLASSES, 2)
            answer = "yes" if (c1 in labs and c2 in labs) else "no"
            pairs.append({
                "patch_id": patch_id,
                "question": f"Does this image contain both {c1.lower()} and {c2.lower()}?",
                "answer": answer,
                "type": "two_class_presence",
            })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bigearthnet", required=True, help="Path to BigEarthNet-MM root")
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument("--out", default="train/vqa_pairs.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    be_root = Path(args.bigearthnet)
    if not be_root.exists():
        print(f"ERROR: BigEarthNet path does not exist: {be_root}")
        print("Download from: https://bigearth.net/")
        return 1

    print(f"Loading BigEarthNet labels from {be_root} (subset={args.subset})…")
    patches = load_bigearthnet_labels(be_root, args.subset)
    if not patches:
        print("ERROR: No label files found. Check the dataset structure.")
        return 1

    rng = random.Random(args.seed)
    pairs = build_pairs(patches, rng)
    rng.shuffle(pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, indent=2))
    print(f"Built {len(pairs)} VQA pairs from {len(patches)} patches → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
