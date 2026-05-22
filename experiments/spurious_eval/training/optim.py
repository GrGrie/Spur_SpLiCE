from __future__ import annotations

import math

import numpy as np
import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper compatible with SpurSSL's call pattern."""

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        sam_no_grad_norm: bool = False,
        only_sam_step_size: bool = False,
        **kwargs,
    ) -> None:
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.sam_no_grad_norm = sam_no_grad_norm
        self.only_sam_step_size = only_sam_step_size
        self.grad_norm_w = None
        self.grad_norm_w_sam = None
        self.grad_w = {}

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        grads = [
            p.grad.detach().to(shared_device).norm(p=2)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not grads:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(grads), p=2)

    @torch.no_grad()
    def first_step(self) -> None:
        grad_norm = self._grad_norm()
        self.grad_norm_w = grad_norm
        scale_norm = torch.tensor(1.0, device=grad_norm.device) if self.sam_no_grad_norm else grad_norm

        for group in self.param_groups:
            scale = group["rho"] / (scale_norm + 1e-12)
            for param in group["params"]:
                if param.grad is None:
                    continue
                delta_w = scale * param.grad
                param.add_(delta_w)
                self.state[param]["delta_w"] = delta_w
                self.grad_w[param] = param.grad.detach().clone()
        self.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                param.sub_(self.state[param]["delta_w"])

        if self.only_sam_step_size:
            self.grad_norm_w_sam = self._grad_norm()
            ratio = (self.grad_norm_w_sam + 1e-12) / (self.grad_norm_w + 1e-12)
            for param in self.grad_w:
                param.grad = self.grad_w[param] * ratio

    @torch.no_grad()
    def step(self, closure=None):
        return self.base_optimizer.step(closure)


def build_optimizer(args, model: torch.nn.Module) -> torch.optim.Optimizer:
    optimizer_name = args.optimizer.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if optimizer_name == "sam":
        base = torch.optim.AdamW if getattr(args, "sam_base_optimizer", "SGD").lower() == "adamw" else torch.optim.SGD
        kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
        if base is torch.optim.SGD:
            kwargs["momentum"] = args.momentum
        return SAM(
            model.parameters(),
            base_optimizer=base,
            rho=args.rho,
            sam_no_grad_norm=args.sam_no_grad_norm,
            only_sam_step_size=args.only_sam_step_size,
            **kwargs,
        )
    raise ValueError(f"Unsupported optimizer '{args.optimizer}'")


def adjust_learning_rate(args, optimizer: torch.optim.Optimizer, epoch: int) -> None:
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate**3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate**steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_learning_rate(args, epoch: int, batch_id: int, total_batches: int, optimizer: torch.optim.Optimizer) -> None:
    if not args.warm or epoch > args.warm_epochs:
        return
    progress = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
    lr = args.warmup_from + progress * (args.warmup_to - args.warmup_from)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
