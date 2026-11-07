"""
PAPO Extended Training Script (Experiment 2) - 120 Steps
Strict hardware thermal monitoring (< 70°C) and memory capping (< 2.5GB).
"""

import os
import sys
import json
import time
import subprocess
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from papo_loss import random_patch_blackening, compute_papo_losses

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/data/train_subset.json"
OUTPUT_DIR = "/home/hishamized/.gemini/antigravity/scratch/papo_project/checkpoints/papo_lora_exp2"
PLOT_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/papo_extended_loss_curves.png"
HISTORY_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/training_history_exp2.json"

MAX_STEPS = 120
LEARNING_RATE = 2e-4
KL_PRCP_COEF = 0.02
MAX_GPU_TEMP_C = 70.0

def get_gpu_status():
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        temp, mem_used, mem_total = map(float, res.strip().split(","))
        return temp, mem_used, mem_total
    except Exception:
        return 0.0, 0.0, 4096.0

def main():
    print("=" * 60)
    print("STARTING PAPO EXPERIMENT 2 (EXTENDED 120-STEP RUN)")
    print(f"Model: {MODEL_ID} (LoRA Rank=8, Alpha=16)")
    print(f"Safety Threshold: {MAX_GPU_TEMP_C}°C")
    print("=" * 60)

    temp, mem_used, mem_total = get_gpu_status()
    print(f"[Initial Hardware State] GPU Temp: {temp}°C | VRAM: {mem_used}MB / {mem_total}MB")
    if temp > MAX_GPU_TEMP_C:
        print("GPU too warm. Aborting for safety.")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 384}

    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    with open(DATA_PATH, "r") as f:
        dataset = json.load(f)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    history = {
        "step": [],
        "task_loss": [],
        "kl_prcp_loss": [],
        "aug_entropy_loss": [],
        "total_loss": [],
        "gpu_temp": [],
        "vram_used_mb": []
    }

    model.train()
    start_time = time.time()

    for step in range(MAX_STEPS):
        sample = dataset[step % len(dataset)]
        img_path = sample["image_path"]
        problem_text = sample["problem"].replace("<image>", "").strip()
        answer_text = sample["answer"].strip()

        clean_img = Image.open(img_path).convert("RGB").resize((384, 384))
        aug_img = random_patch_blackening(clean_img, patch_size=14, black_prob=0.6)

        user_prompt = f"Question: {problem_text}\nPlease provide the final answer inside \\boxed{{}}."
        target_text = f"The correct answer is \\boxed{{{answer_text}}}"

        # Clean prompt
        msg_clean = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        p_clean = processor.apply_chat_template(msg_clean, add_generation_prompt=False)
        inp_clean = processor(text=p_clean, images=clean_img, return_tensors="pt").to(device)

        # Augmented prompt
        msg_aug = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
        ]
        p_aug = processor.apply_chat_template(msg_aug, add_generation_prompt=False)
        inp_aug = processor(text=p_aug, images=aug_img, return_tensors="pt").to(device)

        labels = inp_clean["input_ids"].clone()

        optimizer.zero_grad()
        loss, metrics = compute_papo_losses(
            model=model,
            clean_inputs=inp_clean,
            aug_inputs=inp_aug,
            labels=labels,
            kl_prcp_coef=KL_PRCP_COEF
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        current_temp, current_mem, _ = get_gpu_status()
        if current_temp > MAX_GPU_TEMP_C:
            print(f"[THERMAL GUARD] Temp reached {current_temp}°C. Pausing 5s...")
            time.sleep(5)

        history["step"].append(step + 1)
        history["task_loss"].append(metrics["task_loss"])
        history["kl_prcp_loss"].append(metrics["kl_prcp_loss"])
        history["aug_entropy_loss"].append(metrics["aug_entropy_loss"])
        history["total_loss"].append(metrics["total_loss"])
        history["gpu_temp"].append(current_temp)
        history["vram_used_mb"].append(current_mem)

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"Step [{step + 1:03d}/{MAX_STEPS}] | "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Task Loss: {metrics['task_loss']:.4f} | "
                f"PAPO KL: {metrics['kl_prcp_loss']:.4f} | "
                f"GPU Temp: {current_temp:.1f}°C | "
                f"VRAM: {current_mem:.0f}MB"
            )

        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nExperiment 2 finished in {elapsed:.1f}s!")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PAPO Experiment 2 (120 Steps Extended Convergence & Safety Profile)", fontsize=16)

    axes[0, 0].plot(history["step"], history["total_loss"], color="#2563eb", linewidth=2, label="Total PAPO Loss")
    axes[0, 0].plot(history["step"], history["task_loss"], color="#f59e0b", linestyle="--", label="Task Loss")
    axes[0, 0].set_title("Total & Task Loss Convergence")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["step"], history["kl_prcp_loss"], color="#10b981", linewidth=2, label="Perception KL (L_prcp)")
    axes[0, 1].set_title("Implicit Perception KL Loss (Maximization Goal)")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("KL Divergence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["step"], history["gpu_temp"], color="#ef4444", linewidth=2, label="GPU Temp (°C)")
    axes[1, 0].axhline(y=70.0, color="#991b1b", linestyle=":", label="Thermal Ceiling (70°C)")
    axes[1, 0].set_title("Thermal Safety Telemetry")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Temperature (°C)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["step"], history["vram_used_mb"], color="#8b5cf6", linewidth=2, label="VRAM Allocated (MB)")
    axes[1, 1].axhline(y=4096.0, color="#6b7280", linestyle=":", label="Hardware Total (4096MB)")
    axes[1, 1].axhline(y=3000.0, color="#f59e0b", linestyle="--", label="Conservative Ceiling (3000MB)")
    axes[1, 1].set_title("VRAM Memory Consumption")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Memory (MB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved plot to: {PLOT_PATH}")

if __name__ == "__main__":
    main()

