"""Deterministic full-batch logistic regression with train-only convergence.

An epoch is one L-BFGS step (up to 20 internal iterations). We require ten
consecutive epochs below the gradient tolerance, and retain their predictions
for the last-ten-epoch metric. Evaluation never controls fitting or stopping.
"""
from collections import deque
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass
class LogisticEpoch:
    epoch: int
    objective: float
    gradient_max: float
    train_loss: float
    train_predictions: torch.Tensor
    eval_loss: float
    eval_predictions: torch.Tensor


def fit_logistic_probe(train_x, train_y, eval_x, eval_y, *, num_classes,
                       l2=1e-3, tolerance=1e-6, max_epochs=200):
    """Fit mean CE + l2/2 * ||W||²; do not penalize the intercept.

    Center and standardize each feature using training statistics only. This
    affine transform preserves the linear hypothesis class. CPU float64 avoids
    AMP/feature-scale instability. A finite cap is a failure, not convergence.
    Only the final ten prediction snapshots are retained (bounded memory).
    """
    if not math.isfinite(l2) or not math.isfinite(tolerance) or l2 <= 0 or tolerance <= 0 or max_epochs < 10:
        raise ValueError("Logistic probe needs l2 > 0, tolerance > 0, max_epochs >= 10.")
    x = train_x.detach().cpu().double()
    v = eval_x.detach().cpu().double()
    y = train_y.detach().cpu().long()
    vy = eval_y.detach().cpu().long()
    if (x.ndim != 2 or v.ndim != 2 or x.shape[1] != v.shape[1]
            or y.ndim != 1 or vy.ndim != 1 or len(x) != len(y)
            or len(v) != len(vy) or not len(x) or not len(v)):
        raise ValueError("Invalid logistic feature/target dimensions.")
    if not torch.isfinite(x).all() or not torch.isfinite(v).all():
        raise ValueError("Probe features must be finite.")
    if len(torch.unique(y)) != num_classes:
        raise ValueError("Every probe class must occur in its training split.")
    mean = x.mean(0)
    scale = x.std(0, correction=0)
    scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
    x, v = (x - mean) / scale, (v - mean) / scale
    # Zero initialization needs no random draws and is sufficient for convex LR.
    weight = torch.zeros(num_classes, x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(num_classes, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], lr=1.0, max_iter=20,
                                tolerance_grad=tolerance * 0.1,
                                tolerance_change=1e-15, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        objective = F.cross_entropy(F.linear(x, weight, bias), y) + 0.5 * l2 * weight.square().sum()
        if not torch.isfinite(objective):
            raise RuntimeError("Non-finite logistic training objective.")
        objective.backward()
        return objective

    history = deque(maxlen=10)
    stable = 0
    for epoch in range(1, max_epochs + 1):
        optimizer.step(closure)
        objective = float(closure().detach())
        grad = max(float(weight.grad.abs().max()), float(bias.grad.abs().max()))
        with torch.no_grad():
            train_logits = F.linear(x, weight, bias)
            eval_logits = F.linear(v, weight, bias)
            record = LogisticEpoch(epoch, objective, grad,
                                   float(F.cross_entropy(train_logits, y)), train_logits.argmax(1),
                                   float(F.cross_entropy(eval_logits, vy)), eval_logits.argmax(1))
        history.append(record)
        stable = stable + 1 if grad <= tolerance else 0
        print(f"Logistic epoch {epoch}: objective={objective:.8f}, gradient_max={grad:.3g}, stable={stable}/10", flush=True)
        if stable >= 10:
            return list(history), {
                "converged": True, "epochs": epoch, "gradient_max": grad,
                "objective": objective, "l2": l2, "tolerance": tolerance,
                "averaging_window": 10, "normalization": "train_feature_standardization",
                "solver": "full_batch_lbfgs", "stable_epochs": stable,
            }
    raise RuntimeError(f"Logistic probe did not converge in {max_epochs} epochs: gradient={grad:.3g}, tolerance={tolerance:.3g}. No successful result was recorded.")
