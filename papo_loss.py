"""
PAPO Loss and Perception Grounding Utilities.
Adapted directly from the official PAPO repository (ICLR 2026):
https://github.com/MikeWangWZHL/PAPO
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Literal, Tuple, Dict, Any


def random_patch_blackening(pil_img: Image.Image, patch_size: int = 14, black_prob: float = 0.6) -> Image.Image:
    """
    Randomly blacken square patches in a PIL image.
    Exact implementation from verl/trainer/papo_utils.py in official PAPO.
    """
    img = np.array(pil_img).astype(np.float32)
    h, w = img.shape[:2]
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            if np.random.rand() < black_prob:
                y_end = min(y + patch_size, h)
                x_end = min(x + patch_size, w)
                if img.ndim == 3:
                    img[y:y_end, x:x_end, :] = 0
                else:
                    img[y:y_end, x:x_end] = 0
    return Image.fromarray(img.astype(np.uint8))


def compute_kl(
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    kl_penalty: Literal["kl", "abs", "mse", "low_var_kl", "full"] = "low_var_kl",
) -> torch.Tensor:
    """
    Compute KL divergence between policy log_probs and reference log_probs.
    Exact implementation from verl/trainer/core_algos.py in official PAPO.
    """
    log_probs, ref_log_probs = log_probs.float(), ref_log_probs.float()
    
    if kl_penalty == "kl":
        return log_probs - ref_log_probs

    if kl_penalty == "abs":
        return (log_probs - ref_log_probs).abs()

    if kl_penalty == "mse":
        return 0.5 * (log_probs - ref_log_probs).square()

    if kl_penalty == "low_var_kl":
        # Low-variance KL estimator from Schulman (2020)
        kl = (ref_log_probs - log_probs).clamp(-20.0, 20.0)
        kld = (kl.exp() - kl - 1).contiguous()
        return torch.clamp(kld, min=-10.0, max=10.0)

    if kl_penalty == "full":
        return F.kl_div(ref_log_probs, log_probs, log_target=True, reduction="none").sum(-1)

    raise NotImplementedError(f"Unknown KL penalty: {kl_penalty}")


def compute_papo_losses(
    model,
    clean_inputs: Dict[str, torch.Tensor],
    aug_inputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    kl_prcp_coef: float = 0.02,
    use_aug_entropy: bool = True,
    aug_entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the unified PAPO objective:
    1. Task Supervised Loss (Cross Entropy on Clean Image)
    2. Implicit Perception Loss (KL Divergence between Clean and Blackened Image predictions)
    3. Double Entropy Regularization
    
    Objective:
      Minimizing L_total = L_task - kl_prcp_coef * L_prcp + aug_entropy_coef * L_aug_entropy
      (Minimizing this pushes L_prcp HIGHER, forcing model predictions to be sensitive to visual inputs).
    """
    # 1. Forward pass on Clean Input
    clean_outputs = model(**clean_inputs)
    clean_logits = clean_outputs.logits  # (batch_size, seq_len, vocab_size)

    # Shift tokens for causal language modeling
    shift_logits = clean_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute task loss (cross entropy over valid answer tokens)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    task_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # Compute per-token log probabilities for clean input
    clean_log_probs = F.log_softmax(shift_logits, dim=-1)
    target_clean_log_probs = torch.gather(
        clean_log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)
    
    # Valid response mask
    response_mask = (shift_labels != -100).float()

    # 2. Forward pass on Corrupted (Augmented) Input
    with torch.set_grad_enabled(model.training):
        aug_outputs = model(**aug_inputs)
        aug_logits = aug_outputs.logits
        shift_aug_logits = aug_logits[..., :-1, :].contiguous()
        aug_log_probs = F.log_softmax(shift_aug_logits, dim=-1)
        target_aug_log_probs = torch.gather(
            aug_log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

    # 3. Compute Implicit Perception KL Loss (L_prcp)
    # Measures how much the token probabilities diverge when image is corrupted
    aug_kld = compute_kl(
        log_probs=target_clean_log_probs,
        ref_log_probs=target_aug_log_probs,
        kl_penalty="low_var_kl",
    )
    
    # Masked average over valid response tokens
    valid_tokens = response_mask.sum().clamp(min=1.0)
    kl_prcp_loss = (aug_kld * response_mask).sum() / valid_tokens

    # 4. Double Entropy Regularization
    # Entropy estimator: -mean(log_probs)
    aug_entropy_loss = -((target_aug_log_probs * response_mask).sum() / valid_tokens)

    # 5. Total Unified PAPO Loss
    # We subtract kl_prcp_loss so gradient descent MAXIMIZES the KL divergence!
    total_loss = task_loss - (kl_prcp_coef * kl_prcp_loss)
    if use_aug_entropy:
        total_loss = total_loss + (aug_entropy_coef * aug_entropy_loss)

    metrics = {
        "task_loss": task_loss.item(),
        "kl_prcp_loss": kl_prcp_loss.item(),
        "aug_entropy_loss": aug_entropy_loss.item(),
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics

