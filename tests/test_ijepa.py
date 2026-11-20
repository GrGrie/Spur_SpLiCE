"""The I-JEPA encoder loads an official-format checkpoint and yields one vector per image."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from cospro.models.ijepa import IJEPA_SPECS, IJEPAEncoder, IJEPASpec, build_ijepa_encoder, target_encoder_state

TINY = IJEPASpec("tiny.pth.tar", image_size=28, patch_size=14, embed_dim=8, depth=2, num_heads=2)


def official_checkpoint(encoder: IJEPAEncoder) -> dict:
    """The layout of a facebookresearch/ijepa checkpoint: DDP-prefixed encoders plus training state."""

    prefixed = {f"module.{key}": value for key, value in encoder.state_dict().items()}
    return {"encoder": prefixed, "predictor": {}, "target_encoder": prefixed, "epoch": 300, "lr": 0.001}


class IJEPAEncoderTests(unittest.TestCase):
    def test_official_vith14_shape(self):
        spec = IJEPA_SPECS["ijepa_vith14_in1k"]
        self.assertEqual((spec.image_size // spec.patch_size) ** 2, 256)
        self.assertTrue(spec.url.endswith("IN1K-vit.h.14-300e.pth.tar"))

    def test_target_encoder_loads_with_strict_keys(self):
        torch.manual_seed(0)
        source = IJEPAEncoder(TINY)
        state = target_encoder_state(official_checkpoint(source))
        loaded = IJEPAEncoder(TINY)
        loaded.load_state_dict(state, strict=True)
        images = torch.randn(3, 3, 28, 28)
        features = loaded(images)
        self.assertEqual(features.shape, (3, 8))
        self.assertTrue(torch.allclose(features, source(images)))

    def test_build_reads_a_checkpoint_file(self):
        torch.manual_seed(0)
        source = IJEPAEncoder(TINY)
        IJEPA_SPECS["tiny"] = TINY
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / TINY.filename
                torch.save(official_checkpoint(source), path)
                encoder, feature_dim = build_ijepa_encoder("tiny", path)
        finally:
            del IJEPA_SPECS["tiny"]
        self.assertEqual(feature_dim, 8)
        self.assertTrue(torch.equal(encoder.norm.weight, source.norm.weight))

    def test_wrong_input_size_is_rejected(self):
        with self.assertRaises(ValueError):
            IJEPAEncoder(TINY)(torch.randn(1, 3, 32, 32))

    def test_missing_target_encoder_is_reported(self):
        with self.assertRaises(KeyError):
            target_encoder_state({"encoder": {}})


if __name__ == "__main__":
    unittest.main()
