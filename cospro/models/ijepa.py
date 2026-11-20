"""Frozen I-JEPA encoders (Assran et al., ICCV 2023, arXiv:2301.08243) for the linear probe.

The official checkpoints hold the context encoder, the predictor, the EMA target encoder and the
optimizer state. The probe uses the target encoder, as the paper does for its evaluations, and
average-pools its last-layer patch tokens into one image vector.

The ViT below reproduces the parameter names of ``facebookresearch/ijepa`` (``src/models/
vision_transformer.py``), so the official state dict loads with strict key matching and the
project carries no copy of that repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_BASE_URL = "https://dl.fbaipublicfiles.com/ijepa"


@dataclass(frozen=True)
class IJEPASpec:
    """One official I-JEPA release: its ViT shape and where its checkpoint lives."""

    filename: str
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 16
    mlp_ratio: float = 4.0

    @property
    def url(self) -> str:
        return f"{CHECKPOINT_BASE_URL}/{self.filename}"


IJEPA_SPECS: dict[str, IJEPASpec] = {
    # ViT-H/14 pretrained 300 epochs on ImageNet-1k: 79.3 ImageNet linear-probe top-1 in the paper.
    "ijepa_vith14_in1k": IJEPASpec("IN1K-vit.h.14-300e.pth.tar"),
    # ViT-H/14 pretrained on ImageNet-22k.
    "ijepa_vith14_in22k": IJEPASpec("IN22K-vit.h.14-900e.pth.tar"),
}
IJEPA_MODEL_NAMES = tuple(IJEPA_SPECS)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(batch, tokens, dim))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class IJEPAEncoder(nn.Module):
    """The I-JEPA ViT; ``forward`` returns average-pooled patch tokens of shape [batch, embed_dim]."""

    def __init__(self, spec: IJEPASpec) -> None:
        super().__init__()
        self.spec = spec
        self.patch_embed = PatchEmbed(spec.patch_size, spec.embed_dim)
        num_patches = (spec.image_size // spec.patch_size) ** 2
        # Fixed sin-cos table in the original; the checkpoint supplies its values.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, spec.embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(spec.embed_dim, spec.num_heads, spec.mlp_ratio) for _ in range(spec.depth)])
        self.norm = nn.LayerNorm(spec.embed_dim, eps=1e-6)

    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        expected = (self.spec.image_size, self.spec.image_size)
        if tuple(images.shape[-2:]) != expected:
            raise ValueError(f"I-JEPA {self.spec.filename} expects {expected} inputs; got {tuple(images.shape[-2:])}.")
        x = self.patch_embed(images) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # float16 autocast on CUDA, which the V100 supports; features return as float32.
        with torch.autocast("cuda", dtype=torch.float16, enabled=images.is_cuda):
            tokens = self.patch_tokens(images)
        return tokens.float().mean(dim=1)


def target_encoder_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    """The EMA target encoder of an official checkpoint, without the DistributedDataParallel prefix."""

    if "target_encoder" not in checkpoint:
        raise KeyError(f"Checkpoint has no 'target_encoder'; keys: {sorted(checkpoint)}")
    return {key.removeprefix("module."): value for key, value in checkpoint["target_encoder"].items()}


def default_checkpoint_path(name: str) -> Path:
    """Where the official checkpoint of ``name`` lives on scratch."""

    from cospro.tracking.artifacts import scratch_root

    return scratch_root() / "pretrained" / "ijepa" / IJEPA_SPECS[name].filename


def ensure_checkpoint(name: str, checkpoint: str | Path = "") -> Path:
    """The local checkpoint path, downloading the official file once when none is given."""

    path = Path(checkpoint) if checkpoint else default_checkpoint_path(name)
    if path.exists():
        return path
    if checkpoint:
        raise FileNotFoundError(f"I-JEPA checkpoint not found: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading {IJEPA_SPECS[name].url} to {path} (about 10 GB)")
    partial = path.with_suffix(path.suffix + ".partial")
    torch.hub.download_url_to_file(IJEPA_SPECS[name].url, str(partial), progress=False)
    partial.replace(path)
    return path


def build_ijepa_encoder(name: str, checkpoint: str | Path = "") -> tuple[IJEPAEncoder, int]:
    """The frozen target encoder of the official ``name`` checkpoint and its feature dimension."""

    if name not in IJEPA_SPECS:
        raise ValueError(f"Unknown I-JEPA model {name!r}. Choices: {list(IJEPA_MODEL_NAMES)}")
    spec = IJEPA_SPECS[name]
    path = ensure_checkpoint(name, checkpoint)
    print(f"[INFO] Loading I-JEPA target encoder from {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    encoder = IJEPAEncoder(spec)
    encoder.load_state_dict(target_encoder_state(state), strict=True)
    del state
    return encoder, spec.embed_dim
