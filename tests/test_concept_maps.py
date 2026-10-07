"""Spatial concept maps: window assembly, crop-aware warping, the per-location cross-fit and view boxes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from cospro.data.transforms import RecordedTwoCropTransform
from cospro.methods.concept_factors import dropped_factors, load_concept_maps
from cospro.pipeline.concept_maps import (
    assemble_windows,
    concept_regions,
    region_contrast_loss,
    spatial_cross_fit_loss,
    warp_maps,
    window_offsets,
)


class AssemblyTests(unittest.TestCase):
    def test_windows_tile_and_average_into_a_fourteen_grid(self):
        offsets = window_offsets(448, 112)
        self.assertEqual(offsets, [0, 112, 224])
        ones = assemble_windows(torch.ones(2, 9, 3, 7, 7), 448, 112)
        self.assertEqual(tuple(ones.shape), (2, 3, 14, 14))
        self.assertTrue(torch.allclose(ones, torch.ones_like(ones)))
        corner = torch.zeros(1, 9, 1, 7, 7)
        corner[0, 0] = 1.0  # only the top-left window sees the concept
        assembled = assemble_windows(corner, 448, 112)[0, 0]
        self.assertEqual(float(assembled[0, 0]), 1.0)
        self.assertEqual(float(assembled[-1, -1]), 0.0)

    def test_a_stride_off_the_grid_is_refused(self):
        with self.assertRaises(ValueError):
            window_offsets(448, 100)


class PrincipalMapTests(unittest.TestCase):
    def test_the_control_keeps_the_leading_directions_of_the_patch_embeddings(self):
        from cospro.cli.build_concept_maps import principal_maps

        generator = torch.Generator().manual_seed(4)
        signal = torch.randn(20, 1, 14, 14, generator=generator)
        dense = torch.cat([3 * signal, 0.1 * torch.randn(20, 7, 14, 14, generator=generator)], dim=1).half()
        maps = principal_maps(dense, 2)
        self.assertEqual(tuple(maps.shape), (20, 2, 14, 14))
        correlation = torch.corrcoef(torch.stack([maps[:, 0].flatten().float(), signal.flatten()]))[0, 1]
        self.assertGreater(abs(float(correlation)), 0.99)


class WarpTests(unittest.TestCase):
    def setUp(self):
        self.maps = torch.linspace(0, 1, 14).repeat(14, 1)[None, None].repeat(2, 1, 1, 1)  # rises left to right

    def test_the_whole_image_keeps_the_map_and_a_flip_mirrors_it(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0, 1.0]])
        warped = warp_maps(self.maps, boxes, (7, 7))
        self.assertLess(float(warped[0, 0, 0, 0]), float(warped[0, 0, 0, -1]))
        self.assertTrue(torch.allclose(warped[1], warped[0].flip(-1)))

    def test_a_left_crop_reads_the_left_half(self):
        boxes = torch.tensor([[0.0, 0.0, 1.0, 0.5, 0.0], [0.0, 0.5, 1.0, 0.5, 0.0]])
        warped = warp_maps(self.maps, boxes, (7, 7))
        self.assertLess(float(warped[0].max()), 0.55)
        self.assertGreater(float(warped[1].min()), 0.45)


class SpatialLossTests(unittest.TestCase):
    def test_features_that_encode_each_location_explain_it_and_random_ones_do_not(self):
        generator = torch.Generator().manual_seed(0)
        targets = torch.randn(32, 3, 7, 7, generator=generator)
        encoded = torch.cat([targets, 0.05 * torch.randn(32, 13, 7, 7, generator=generator)], dim=1) + 3.0
        weights = torch.ones(32)
        _, explained = spatial_cross_fit_loss(encoded, targets, weights, ridge=0.1)
        self.assertGreater(explained, 0.8)
        _, random_explained = spatial_cross_fit_loss(torch.randn(32, 16, 7, 7, generator=generator), targets,
                                                     weights, ridge=0.1)
        self.assertLess(random_explained, 0.1)

    def test_views_with_zero_weight_do_not_enter_the_loss(self):
        generator = torch.Generator().manual_seed(1)
        features = torch.randn(16, 8, 4, 4, generator=generator, requires_grad=True)
        targets = torch.randn(16, 2, 4, 4, generator=generator)
        weights = torch.ones(16)
        weights[[3, 11]] = 0.0  # both views of image 3
        loss, _ = spatial_cross_fit_loss(features, targets, weights, ridge=1.0)
        loss.backward()
        self.assertEqual(float(features.grad[3].abs().sum()), 0.0)
        self.assertEqual(float(features.grad[11].abs().sum()), 0.0)


class RegionTests(unittest.TestCase):
    def test_a_region_pools_the_features_under_its_concept(self):
        features = torch.zeros(2, 3, 4, 4)
        features[:, 0, :, :2] = 1.0  # channel 0 on the left half
        features[:, 1, :, 2:] = 1.0  # channel 1 on the right half
        probabilities = torch.zeros(2, 3, 4, 4)
        probabilities[:, 0, :, :2] = 1.0  # concept 0 on the left, concept 2 on the right
        probabilities[:, 2, :, 2:] = 1.0
        pooled, concepts, views, _ = concept_regions(features, probabilities, torch.ones(2), min_mass=0.1, top=2)
        self.assertEqual(len(pooled), 4)
        for vector, concept in zip(pooled, concepts.tolist()):
            expected = torch.tensor([1.0, 0.0, 0.0]) if concept == 0 else torch.tensor([0.0, 1.0, 0.0])
            self.assertTrue(torch.allclose(vector, expected))
        _, _, kept_views, _ = concept_regions(features, probabilities, torch.tensor([1.0, 0.0]), min_mass=0.1, top=2)
        self.assertEqual(set(kept_views.tolist()), {0})

    def test_aligned_concepts_across_images_lower_the_loss(self):
        generator = torch.Generator().manual_seed(5)
        concepts = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        views = torch.arange(8)  # eight views of eight images
        masses = torch.rand(8, 2, generator=generator)
        prototypes = F.normalize(torch.randn(2, 16, generator=generator), dim=1)
        aligned = F.normalize(prototypes[concepts] + 0.05 * torch.randn(8, 16, generator=generator), dim=1)
        scattered = F.normalize(torch.randn(8, 16, generator=generator), dim=1)
        low, diagnostics = region_contrast_loss(aligned, concepts, views, masses, 8, 0.1, 1.0)
        high, _ = region_contrast_loss(scattered, concepts, views, masses, 8, 0.1, 1.0)
        self.assertLess(float(low), float(high))
        self.assertEqual(diagnostics["factor_region_anchor_fraction"], 1.0)

    def test_regions_of_one_image_are_never_positives(self):
        concepts = torch.tensor([0, 0])
        views = torch.tensor([0, 4])  # both views of image 0 when the batch holds four images
        loss, diagnostics = region_contrast_loss(F.normalize(torch.randn(2, 4), dim=1), concepts, views,
                                                 torch.rand(8, 3), 4, 0.1, 1.0)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(diagnostics["factor_region_anchor_fraction"], 0.0)


class ViewBoxTests(unittest.TestCase):
    def test_the_recorded_box_and_flip_reproduce_each_view(self):
        array = np.random.default_rng(0).integers(0, 255, (60, 90, 3), dtype=np.uint8)
        image = Image.fromarray(array)
        pipeline = transforms.Compose([transforms.RandomResizedCrop(24, scale=(0.3, 1.0)),
                                       transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        recorded = RecordedTwoCropTransform(pipeline)
        torch.manual_seed(3)
        for _ in range(5):
            views, boxes = recorded(image)
            for view, (top, left, height, width, flip) in zip(views, boxes.tolist()):
                crop = TF.resized_crop(image, round(top * 60), round(left * 90), round(height * 60),
                                       round(width * 90), [24, 24], antialias=True)
                expected = TF.to_tensor(TF.hflip(crop) if flip else crop)
                self.assertTrue(torch.allclose(view, expected))


class LoadMapsTests(unittest.TestCase):
    def test_maps_follow_the_subset_order_and_refuse_other_factors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.pt"
            maps = torch.rand(4, 2, 14, 14).half()
            torch.save({"sample_ids": [f"waterbirds:{index}" for index in (5, 6, 7, 8)], "factor_names": ["a", "b"],
                        "maps": maps}, path)
            loaded = load_concept_maps(str(path), ["a", "b"], "waterbirds", [7, 5], shuffled=False)
            self.assertEqual(tuple(loaded.shape), (2, 2, 14, 14))
            standardized = (maps.float() - maps.float().mean(dim=(0, 2, 3), keepdim=True)) / maps.float().std(
                dim=(0, 2, 3), keepdim=True)
            self.assertTrue(torch.allclose(loaded[0].float(), standardized[2], atol=1e-2))
            with self.assertRaises(ValueError):
                load_concept_maps(str(path), ["a", "c"], "waterbirds", [7, 5], shuffled=False)

    def test_dropped_concepts_leave_the_maps(self):
        names = ["kitty | kitten | cat", "couch", "canine | dog", "pups | pup | puppies (+2)", "catalog"]
        self.assertEqual(dropped_factors(names, "cat, puppies"), [0, 3])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "maps.pt"
            torch.save({"sample_ids": ["waterbirds:1", "waterbirds:2"], "factor_names": names,
                        "maps": torch.rand(2, 5, 14, 14).half()}, path)
            loaded = load_concept_maps(str(path), names, "waterbirds", [1, 2], shuffled=False, drop="couch")
            self.assertEqual(loaded.shape[1], 4)


if __name__ == "__main__":
    unittest.main()
