"""
PAPO Evaluation Script for Validation Benchmark (MMK12).
Evaluates both the original Base Model and the PAPO fine-tuned model on:
1. Ground Truth Accuracy (Math / Visual Reasoning outcome)
2. Perception Grounding Sensitivity (divergence when image is masked)
"""

import os
import re
import json
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from papo_loss import random_patch_blackening, compute_kl

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHECKPOINT_DIR = "/home/hishamized/.gemini/antigravity/scratch/papo_project/checkpoints/papo_lora_model"
VAL_DATA_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/data/val_subset.json"
RESULTS_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/evaluation_results.json"

def extract_answer(text: str) -> str:
    """Extracts answer inside boxed{} or last option."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    opt_match = re.findall(r"\b([A-D])\b", text)
    if opt_match:
        return opt_match[-1]
    return text.strip()

def evaluate_model(model, processor, dataset, device, num_samples=15):
    results = []
    correct_count = 0
    total_perception_kl = 0.0

    model.eval()
    for idx in range(min(num_samples, len(dataset))):
        sample = dataset[idx]
        img_path = sample["image_path"]
        problem_text = sample["problem"].replace("<image>", "").strip()
        ground_truth = sample["answer"].strip()

        clean_img = Image.open(img_path).convert("RGB").resize((384, 384))
        aug_img = random_patch_blackening(clean_img, patch_size=14, black_prob=0.6)

        user_prompt = f"Question: {problem_text}\nPlease provide the final answer inside \\boxed{{}}."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs_clean = processor(text=prompt, images=clean_img, return_tensors="pt").to(device)
        inputs_aug = processor(text=prompt, images=aug_img, return_tensors="pt").to(device)

        with torch.no_grad():
            # 1. Generate prediction with clean image
            gen_ids = model.generate(
                **inputs_clean,
                max_new_tokens=48,
                do_sample=False,
                temperature=None,
                top_p=None
            )
            gen_tokens = gen_ids[0][inputs_clean["input_ids"].shape[1]:]
            pred_text = processor.decode(gen_tokens, skip_special_tokens=True).strip()

            # 2. Forward pass to measure perception sensitivity (KL clean vs masked)
            clean_out = model(**inputs_clean)
            aug_out = model(**inputs_aug)
            
            clean_lp = torch.log_softmax(clean_out.logits[0, -1, :], dim=-1)
            aug_lp = torch.log_softmax(aug_out.logits[0, -1, :], dim=-1)
            prcp_kl = compute_kl(clean_lp, aug_lp, kl_penalty="low_var_kl").mean().item()

        pred_ans = extract_answer(pred_text)
        is_correct = (pred_ans.lower() == ground_truth.lower())
        if is_correct:
            correct_count += 1
        total_perception_kl += prcp_kl

        results.append({
            "id": sample["id"],
            "ground_truth": ground_truth,
            "prediction": pred_ans,
            "full_response": pred_text,
            "is_correct": is_correct,
            "perception_kl": prcp_kl
        })

    accuracy = correct_count / len(results)
    avg_perception_kl = total_perception_kl / len(results)
    return accuracy, avg_perception_kl, results

def main():
    print("=" * 60)
    print("RUNNING PAPO VALIDATION EVALUATION")
    print(f"Dataset: {VAL_DATA_PATH}")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(VAL_DATA_PATH, "r") as f:
        val_dataset = json.load(f)

    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 384}

    # 1. Evaluate Base Model
    print("\n[1/2] Evaluating Base Model...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)
    base_acc, base_kl, base_details = evaluate_model(base_model, processor, val_dataset, device, num_samples=15)
    print(f"Base Model Accuracy: {base_acc * 100:.1f}% | Avg Perception KL: {base_kl:.4f}")

    # 2. Evaluate PAPO LoRA Model
    print("\n[2/2] Evaluating PAPO-Tuned Model...")
    papo_model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    papo_acc, papo_kl, papo_details = evaluate_model(papo_model, processor, val_dataset, device, num_samples=15)
    print(f"PAPO Model Accuracy: {papo_acc * 100:.1f}% | Avg Perception KL: {papo_kl:.4f}")

    # Compile Final Benchmark Summary
    summary = {
        "benchmark": "PAPOGalaxy/PAPO_MMK12_test",
        "evaluation_samples": len(base_details),
        "base_model": {
            "name": BASE_MODEL_ID,
            "accuracy": base_acc,
            "avg_perception_kl": base_kl
        },
        "papo_model": {
            "name": "SmolVLM-500M + PAPO LoRA",
            "accuracy": papo_acc,
            "avg_perception_kl": papo_kl
        },
        "perception_grounding_gain": papo_kl - base_kl,
        "sample_comparisons": [
            {
                "id": b["id"],
                "ground_truth": b["ground_truth"],
                "base_pred": b["prediction"],
                "papo_pred": p["prediction"],
                "base_correct": b["is_correct"],
                "papo_correct": p["is_correct"],
                "base_kl": b["perception_kl"],
                "papo_kl": p["perception_kl"]
            }
            for b, p in zip(base_details, papo_details)
        ]
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full evaluation summary to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
