import json
import torch
import re
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from papo_loss import random_patch_blackening, compute_kl

BASE_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
EXP1_DIR = "checkpoints/papo_lora_model"
EXP2_DIR = "checkpoints/papo_lora_exp2"
VAL_PATH = "data/val_subset.json"
OUT_PATH = "dual_experiment_comparison.json"

def extract_ans(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()
    opts = re.findall(r"\b([A-D])\b", text)
    if opts:
        return opts[-1]
    return text.strip()[:100]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
processor.image_processor.do_image_splitting = False
processor.image_processor.size = {"longest_edge": 384}

base_model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL_ID, dtype=torch.float16).to(device)

with open(VAL_PATH) as f:
    val_data = json.load(f)[:10]

print("Evaluating Base Model...")
base_preds = []
for item in val_data:
    img = Image.open(item["image_path"]).convert("RGB").resize((384, 384))
    prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {item['problem'].replace('<image>', '').strip()}\nFinal answer inside \\boxed{{}}."}]}], add_generation_prompt=True)
    inp = processor(text=prompt, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = base_model.generate(**inp, max_new_tokens=40, do_sample=False)
    pred_txt = processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    base_preds.append({"pred": extract_ans(pred_txt), "raw": pred_txt})

print("Evaluating Exp 1 Model...")
exp1_model = PeftModel.from_pretrained(base_model, EXP1_DIR)
exp1_preds = []
for item in val_data:
    img = Image.open(item["image_path"]).convert("RGB").resize((384, 384))
    prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {item['problem'].replace('<image>', '').strip()}\nFinal answer inside \\boxed{{}}."}]}], add_generation_prompt=True)
    inp = processor(text=prompt, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = exp1_model.generate(**inp, max_new_tokens=40, do_sample=False)
    pred_txt = processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    exp1_preds.append({"pred": extract_ans(pred_txt), "raw": pred_txt})

print("Evaluating Exp 2 Model...")
exp2_model = PeftModel.from_pretrained(base_model, EXP2_DIR)
exp2_preds = []
for item in val_data:
    img = Image.open(item["image_path"]).convert("RGB").resize((384, 384))
    prompt = processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Question: {item['problem'].replace('<image>', '').strip()}\nFinal answer inside \\boxed{{}}."}]}], add_generation_prompt=True)
    inp = processor(text=prompt, images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = exp2_model.generate(**inp, max_new_tokens=40, do_sample=False)
    pred_txt = processor.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    exp2_preds.append({"pred": extract_ans(pred_txt), "raw": pred_txt})

samples = []
for i in range(len(val_data)):
    gt = val_data[i]["answer"].strip()
    samples.append({
        "id": val_data[i]["id"],
        "question": val_data[i]["problem"].replace("<image>", "").strip()[:160],
        "image_path": val_data[i]["image_path"],
        "ground_truth": gt,
        "base_pred": base_preds[i]["pred"],
        "exp1_pred": exp1_preds[i]["pred"],
        "exp2_pred": exp2_preds[i]["pred"],
        "base_correct": base_preds[i]["pred"].lower() == gt.lower(),
        "exp1_correct": exp1_preds[i]["pred"].lower() == gt.lower(),
        "exp2_correct": exp2_preds[i]["pred"].lower() == gt.lower(),
    })

comparison = {
    "samples": samples,
    "accuracy": {
        "base": sum(s["base_correct"] for s in samples) / len(samples),
        "exp1": sum(s["exp1_correct"] for s in samples) / len(samples),
        "exp2": sum(s["exp2_correct"] for s in samples) / len(samples),
    }
}

with open(OUT_PATH, "w") as f:
    json.dump(comparison, f, indent=2)
print("Dual comparison finished successfully! Accuracies:", comparison["accuracy"])
