 """
PAPO Scaled-Down Training Script for Local Laptop (RTX 3050, 4GB VRAM).
Preserves the full mathematical formulation of PAPO:
- Implicit Perception KL Loss (L_prcp) via Random Patch Blackening
- Double Entropy Regularization
- Conservative hardware thermal & memory controls
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

# Configuration
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DATA_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/data/train_subset.json"
OUTPUT_DIR = "/home/hishamized/.gemini/antigravity/scratch/papo_project/checkpoints/papo_lora_model"
PLOT_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/papo_training_loss_curves.png"
HISTORY_PATH = "/home/hishamized/.gemini/antigravity/scratch/papo_project/training_history.json"

MAX_STEPS = 40
LEARNING_RATE = 2e-4
KL_PRCP_COEF = 0.02
MAX_GPU_TEMP_C = 72.0
MAX_VRAM_MB = 3200.0  # Conservative 3.2GB limit (out of 4GB)

def get_gpu_status():
    """Queries NVIDIA-SMI for current GPU temperature and memory usage."""
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        temp, mem_used, mem_total = map(float, res.strip().split(","))
        return temp, mem_used, mem_total
    except Exception as e:
        return 0.0, 0.0, 4096.0

def main():
    print("=" * 60)
    print("STARTING PAPO SCALED-DOWN EXPERIMENT")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {DATA_PATH}")
    print(f"Perception KL Coef: {KL_PRCP_COEF}")
    print("=" * 60)

    # Hardware safety pre-check
    temp, mem_used, mem_total = get_gpu_status()
    print(f"[Hardware Pre-Check] GPU Temp: {temp}°C | VRAM Used: {mem_used}MB / {mem_total}MB")
    if temp > MAX_GPU_TEMP_C:
        print(f"ERROR: GPU temperature ({temp}°C) exceeds safety limit ({MAX_GPU_TEMP_C}°C). Aborting.")
        sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Processor and optimize for 4GB VRAM
    print("Loading processor with memory-efficient single-scale configuration...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    processor.image_processor.size = {"longest_edge": 384}

    # Load Model in float16
    print("Loading base model in float16...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    # Apply LoRA for parameter-efficient adaptation
    print("Configuring LoRA adapters...")
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

    # Load dataset
    with open(DATA_PATH, "r") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} training examples from {DATA_PATH}")

    # Optimizer
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
    print("\nStarting training loop...")
    start_time = time.time()

    for step in range(MAX_STEPS):
        sample = dataset[step % len(dataset)]
        img_path = sample["image_path"]
        problem_text = sample["problem"].replace("<image>", "").strip()
        answer_text = sample["answer"].strip()

        # Load clean image and resize cleanly
        clean_img = Image.open(img_path).convert("RGB").resize((384, 384))
        # Generate corrupted image via PAPO Random Patch Blackening
        aug_img = random_patch_blackening(clean_img, patch_size=14, black_prob=0.6)

        # Build prompt using standard instruction format
        user_prompt = f"Question: {problem_text}\nPlease provide the final answer inside \\boxed{{}}."
        target_text = f"The correct answer is \\boxed{{{answer_text}}}"

        # Prepare Clean Inputs
        messages_clean = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text}
                ]
            }
        ]
        prompt_clean = processor.apply_chat_template(messages_clean, add_generation_prompt=False)
        inputs_clean = processor(
            text=prompt_clean,
            images=clean_img,
            return_tensors="pt"
        ).to(device)

        # Prepare Augmented Inputs (Corrupted Image)
        messages_aug = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": target_text}
                ]
            }
        ]
        prompt_aug = processor.apply_chat_template(messages_aug, add_generation_prompt=False)
        inputs_aug = processor(
            text=prompt_aug,
            images=aug_img,
            return_tensors="pt"
        ).to(device)

        # Prepare labels
        labels = inputs_clean["input_ids"].clone()

        # Forward & Loss Computation
        optimizer.zero_grad()
        loss, metrics = compute_papo_losses(
            model=model,
            clean_inputs=inputs_clean,
            aug_inputs=inputs_aug,
            labels=labels,
            kl_prcp_coef=KL_PRCP_COEF
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Hardware safety check
        current_temp, current_mem, _ = get_gpu_status()
        if current_temp > MAX_GPU_TEMP_C:
            print(f"[SAFETY ALERT] GPU Temp reached {current_temp}°C! Pausing 10s to cool down...")
            time.sleep(10)
            current_temp, _, _ = get_gpu_status()
            if current_temp > MAX_GPU_TEMP_C:
                print(f"[SAFETY STOP] GPU Temp still {current_temp}°C. Stopping training early safely.")
                break

        # Log metrics
        history["step"].append(step + 1)
        history["task_loss"].append(metrics["task_loss"])
        history["kl_prcp_loss"].append(metrics["kl_prcp_loss"])
        history["aug_entropy_loss"].append(metrics["aug_entropy_loss"])
        history["total_loss"].append(metrics["total_loss"])
        history["gpu_temp"].append(current_temp)
        history["vram_used_mb"].append(current_mem)

        if (step + 1) % 5 == 0 or step == 0:
            print(
                f"Step [{step + 1:02d}/{MAX_STEPS}] | "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Task Loss: {metrics['task_loss']:.4f} | "
                f"PAPO KL: {metrics['kl_prcp_loss']:.4f} | "
                f"GPU Temp: {current_temp:.1f}°C | "
                f"VRAM: {current_mem:.0f}MB"
            )

        # Clear cache to keep memory clean
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"\nTraining completed successfully in {elapsed:.1f} seconds!")

    # Save Checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Saved trained PAPO LoRA model to: {OUTPUT_DIR}")

    # Save History JSON
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to: {HISTORY_PATH}")

    # Generate Training Curves Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PAPO Scaled Experiment: Training Dynamics & Hardware Safety", fontsize=16)

    # 1. Total Loss & Task Loss
    axes[0, 0].plot(history["step"], history["total_loss"], label="Total PAPO Loss", color="blue", linewidth=2)
    axes[0, 0].plot(history["step"], history["task_loss"], label="Task Loss", color="orange", linestyle="--")
    axes[0, 0].set_title("Loss Convergence")
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. PAPO Perception KL Divergence
    axes[0, 1].plot(history["step"], history["kl_prcp_loss"], label="Perception KL (L_prcp)", color="green", linewidth=2)
    axes[0, 1].set_title("Implicit Perception KL Divergence (Sensitivity to Masking)")
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("KL Divergence")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. GPU Temperature
    axes[1, 0].plot(history["step"], history["gpu_temp"], label="GPU Temp (°C)", color="red", linewidth=2)
    axes[1, 0].axhline(y=75.0, color="darkred", linestyle=":", label="Safety Limit (75°C)")
    axes[1, 0].set_title("GPU Temperature Profile (Thermal Safety)")
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Temperature (°C)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. GPU VRAM Memory Usage
    axes[1, 1].plot(history["step"], history["vram_used_mb"], label="VRAM Allocated (MB)", color="purple", linewidth=2)
    axes[1, 1].axhline(y=4096.0, color="gray", linestyle=":", label="Physical VRAM (4096MB)")
    axes[1, 1].axhline(y=3072.0, color="orange", linestyle="--", label="Target Cap (3072MB)")
    axes[1, 1].set_title("VRAM Consumption (Memory Safety)")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Memory (MB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved training plot to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
