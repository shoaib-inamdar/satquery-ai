"""
train/lora_bigearthnet.py
LoRA fine-tune of the VLM on BigEarthNet VQA pairs.
Saves adapter to models/adapters/bigearthnet_lora/.
Writes train_report.json with subset size, steps, and loss.

Usage (run separately — do NOT run automatically):
    python train/lora_bigearthnet.py \\
        --pairs train/vqa_pairs.json \\
        --model HuggingFaceTB/SmolVLM-256M-Instruct \\
        --epochs 1 \\
        --max_samples 500
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="train/vqa_pairs.json")
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--out", default="models/adapters/bigearthnet_lora")
    args = parser.parse_args()

    # ── Load VQA pairs ────────────────────────────────────────────────────────
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        print(f"ERROR: VQA pairs file not found: {pairs_path}")
        print("Run train/build_vqa_from_bigearthnet.py first.")
        return 1

    pairs = json.loads(pairs_path.read_text())[:args.max_samples]
    print(f"Loaded {len(pairs)} VQA pairs.")

    # ── Load model and tokenizer ───────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from peft import LoraConfig, get_peft_model, TaskType
        from torch.optim import AdamW
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install: pip install torch transformers peft accelerate")
        return 1

    print(f"Loading model {args.model}…")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(args.model)
    base_model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
    ).to(device)

    # ── Apply LoRA ────────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.train()

    # ── Simple training loop (text-only, no images) ───────────────────────────
    losses = []
    t0 = time.time()
    steps = 0

    for epoch in range(args.epochs):
        for pair in pairs:
            prompt = f"Question: {pair['question']} Answer:"
            target = pair["answer"]
            full = prompt + " " + target

            enc = processor.tokenizer(
                full, return_tensors="pt", truncation=True, max_length=128
            ).to(device)
            labels = enc["input_ids"].clone()
            # Mask prompt tokens
            prompt_len = len(processor.tokenizer(prompt)["input_ids"])
            labels[:, :prompt_len] = -100

            outputs = model(**enc, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.item()))
            steps += 1

            if steps % 50 == 0:
                avg_loss = sum(losses[-50:]) / min(50, len(losses))
                print(f"Epoch {epoch+1}/{args.epochs} Step {steps} Loss {avg_loss:.4f}")

    # ── Save adapter ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    processor.save_pretrained(out_path)

    # ── Write report ──────────────────────────────────────────────────────────
    report = {
        "model": args.model,
        "subset_size": len(pairs),
        "epochs": args.epochs,
        "steps": steps,
        "final_loss": round(losses[-1], 6) if losses else None,
        "avg_loss": round(sum(losses) / max(len(losses), 1), 6),
        "duration_s": round(time.time() - t0, 1),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.lr,
        "adapter_path": str(out_path),
    }
    report_path = out_path / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nAdapter saved to {out_path}")
    print(f"Report: {report_path}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
