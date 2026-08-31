import unittest
from unittest import mock

import torch
from torchvision import models

from experiments.spurious_eval.models import resnet as resnet_module


class ResNetModelTests(unittest.TestCase):
    def test_resnet50_pretrained_uses_imagenet_v2_weights_and_large_input_stem(self):
        torchvision_encoder = models.resnet50(weights=None)

        with mock.patch.object(
            resnet_module.models,
            "resnet50",
            return_value=torchvision_encoder,
        ) as resnet50_factory:
            encoder, feature_dim = resnet_module.build_resnet_encoder("resnet50_pretrained")

        resnet50_factory.assert_called_once_with(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.assertEqual(feature_dim, 2048)
        self.assertIsInstance(encoder.fc, torch.nn.Identity)

        resnet18_large, _ = resnet_module.build_resnet_encoder("resnet18_large")
        self.assertEqual(encoder.conv1.kernel_size, resnet18_large.conv1.kernel_size)
        self.assertEqual(encoder.conv1.kernel_size, (7, 7))
        self.assertEqual(encoder.conv1.stride, resnet18_large.conv1.stride)
        self.assertEqual(encoder.conv1.stride, (2, 2))
        self.assertEqual(encoder.maxpool.kernel_size, resnet18_large.maxpool.kernel_size)
        self.assertEqual(encoder.maxpool.kernel_size, 3)
        self.assertEqual(encoder.maxpool.stride, resnet18_large.maxpool.stride)
        self.assertEqual(encoder.maxpool.stride, 2)
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.parameters()))

        encoder.eval()
        with torch.no_grad():
            features = encoder(torch.randn(1, 3, 64, 64))
        self.assertEqual(features.shape, (1, 2048))

    def test_resnet50_checkpoint_loading_can_skip_the_imagenet_download(self):
        torchvision_encoder = models.resnet50(weights=None)

        with mock.patch.object(
            resnet_module.models,
            "resnet50",
            return_value=torchvision_encoder,
        ) as resnet50_factory:
            encoder, feature_dim = resnet_module.build_resnet_encoder(
                "resnet50_pretrained",
                load_pretrained_weights=False,
            )

        resnet50_factory.assert_called_once_with(weights=None)
        self.assertEqual(feature_dim, 2048)
        self.assertIsInstance(encoder.fc, torch.nn.Identity)


if __name__ == "__main__":
    unittest.main()
