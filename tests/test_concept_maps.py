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
from cospro.methods.concept_factors import (
    SpatialConceptDistillation,
    dropped_factors,
    factor_type_scores,
    load_concept_maps,
)
from cospro.pipeline.concept_maps import (
    assemble_windows,
    box_masks,
    composite_images,
    composite_maps,
    concept_regions,
    masked_pool,
    mixed_source_loss,
    object_columns,
    region_contrast_loss,
    sharpen,
    spatial_cross_fit_loss,
    two_view_folds,
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


class CompositeTests(unittest.TestCase):
    def test_the_more_object_like_half_of_the_factors_are_objects(self):
        self.assertEqual(object_columns([-0.2, 0.1, -0.1, 0.0]).tolist(), [True, False, True, False])

    def test_a_composite_keeps_its_region_and_paints_over_the_donor_objects(self):
        image = torch.ones(1, 3, 32, 32)
        donor = torch.zeros(1, 3, 32, 32)
        donor_objects = torch.zeros(1, 1, 32, 32)
        donor_objects[..., 12:20, 20:28] = 1.0
        donor[..., 12:20, 20:28] = 5.0  # the donor's object
        keep = torch.zeros(1, 1, 32, 32)
        keep[..., :, :16] = 1.0
        composite = composite_images(image, donor, keep, donor_objects)
        self.assertTrue(torch.equal(composite[..., :16], image[..., :16]))
        self.assertLess(float(composite[..., 16:].abs().max()), 1e-5)

    def test_composite_maps_follow_the_pixels_and_commute_with_standardization(self):
        maps, donor_maps = torch.rand(2, 5, 7, 7), torch.rand(2, 5, 7, 7)
        keep, donor_objects = torch.rand(2, 1, 7, 7), torch.rand(2, 1, 7, 7)
        self.assertTrue(torch.allclose(composite_maps(maps, donor_maps, torch.ones_like(keep), donor_objects), maps))
        self.assertTrue(torch.allclose(
            composite_maps(maps, donor_maps, torch.zeros_like(keep), torch.zeros_like(donor_objects)), donor_maps))
        mean, std = torch.rand(1, 5, 1, 1), torch.rand(1, 5, 1, 1) + 0.5
        standardized = composite_maps((maps - mean) / std, (donor_maps - mean) / std, keep, donor_objects)
        self.assertTrue(torch.allclose(standardized, (composite_maps(maps, donor_maps, keep, donor_objects) - mean) / std,
                                       atol=1e-5))

    def _distillation(self, mask: str) -> SpatialConceptDistillation:
        # Factor 0 is the object and fills the left half of every image; factor 1 fills the right half.
        probabilities = torch.zeros(4, 2, 14, 14)
        probabilities[:, 0, :, :7] = 1.0
        probabilities[:, 1, :, 7:] = 1.0
        distillation = SpatialConceptDistillation(
            probabilities.clone(), 1.0, 0, 0, sample_weights=torch.tensor([1.0, 1.0, 1.0, 0.0]), ridge=1.0,
            probabilities=probabilities, composite={"fraction": 1.0, "mask": mask, "min_mass": 0.05},
            objects=torch.tensor([True, False]),
        )
        distillation.set_epoch(1)
        return distillation

    def _batch(self):
        views = [torch.rand(4, 3, 28, 28), torch.rand(4, 3, 28, 28)]
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]).repeat(4, 2, 1)
        return views, (views, torch.arange(4), boxes)

    def test_compositing_pastes_the_object_half_and_leaves_held_out_images_alone(self):
        distillation = self._distillation("object")
        views, batch = self._batch()
        composed = distillation.compose([view.clone() for view in views], batch)
        plan = distillation.plan
        self.assertEqual(plan["images"].tolist(), [0, 1, 2])
        self.assertNotIn(3, plan["donors"].tolist())
        self.assertFalse(bool((plan["donors"] == plan["images"]).any()))
        self.assertTrue(torch.equal(composed[0], views[0]))
        self.assertTrue(torch.equal(composed[1][3], views[1][3]))
        self.assertTrue(torch.allclose(composed[1][:3, :, :, :12], views[1][:3, :, :, :12]))
        self.assertFalse(torch.allclose(composed[1][:3, :, :, 16:], views[1][:3, :, :, 16:]))

    def test_the_context_control_keeps_the_other_half(self):
        distillation = self._distillation("context")
        views, batch = self._batch()
        composed = distillation.compose([view.clone() for view in views], batch)
        self.assertTrue(torch.allclose(composed[1][:3, :, :, 16:], views[1][:3, :, :, 16:]))

    def test_composited_targets_match_the_composite_maps(self):
        distillation = self._distillation("object")
        views, batch = self._batch()
        distillation.compose([view.clone() for view in views], batch)
        plan = distillation.plan
        boxes = torch.cat([batch[2][:, 0], batch[2][:, 1]])
        warped = torch.rand(8, 2, 7, 7)
        composited = distillation._composited(warped, plan, boxes, (7, 7))
        images, donors = 4 + plan["images"], 4 + plan["donors"]
        keep = sharpen(warp_maps(plan["keep"], boxes[images], (7, 7)))
        donor_objects = sharpen(warp_maps(plan["donor_objects"], boxes[donors], (7, 7)))
        self.assertTrue(torch.allclose(composited[images], composite_maps(warped[images], warped[donors], keep,
                                                                          donor_objects)))
        self.assertTrue(torch.equal(composited[:4], warped[:4]))

    def test_type_scores_must_belong_to_the_concept_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = Path(directory) / "concept_groups.json"
            groups.write_text("{}", encoding="utf-8")
            from cospro.pipeline.concept_type import concept_groups_digest
            scores = Path(directory) / "scores.json"
            scores.write_text(f'{{"concept_groups_sha256": "{concept_groups_digest(groups)}", "scores": {{"1": -0.2, "2": 0.1}}}}',
                              encoding="utf-8")
            factors = [{"group_ids": [1]}, {"group_ids": [1, 2]}]
            self.assertTrue(np.allclose(factor_type_scores(factors, str(scores), groups), [-0.2, -0.05]))
            groups.write_text("{ }", encoding="utf-8")
            with self.assertRaises(ValueError):
                factor_type_scores(factors, str(scores), groups)


class MixTests(unittest.TestCase):
    def test_a_box_covers_its_area_at_every_resolution(self):
        boxes = torch.tensor([[0.1, 0.2, 0.5, 0.4], [0.0, 0.0, 1.0, 1.0]])
        for size in [(7, 7), (28, 28), (5, 9)]:
            masks = box_masks(boxes, size)
            self.assertEqual(tuple(masks.shape), (2, 1, *size))
            self.assertTrue(torch.allclose(masks.mean(dim=(1, 2, 3)), torch.tensor([0.2, 1.0]), atol=1e-5))
        self.assertTrue(torch.allclose(box_masks(boxes[1:], (7, 7)), torch.ones(1, 1, 7, 7)))

    def test_masked_pooling_averages_inside_the_mask(self):
        features = torch.arange(8.0).view(1, 2, 2, 2)
        mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        self.assertTrue(torch.allclose(masked_pool(features, mask), torch.tensor([[1.5, 5.5]])))

    def test_one_source_is_infonce_and_two_sources_mix_the_targets(self):
        queries = F.normalize(torch.randn(3, 8), dim=1)
        keys = F.normalize(torch.randn(5, 8), dim=1)
        sources = torch.tensor([[1], [4], [0]])
        hard = mixed_source_loss(queries, keys, sources, torch.ones(3, 1), 0.1)
        self.assertTrue(torch.allclose(hard, F.cross_entropy(queries @ keys.T / 0.1, sources[:, 0])))
        pair = torch.tensor([[1, 2]])
        soft = mixed_source_loss(queries[:1], keys, pair, torch.tensor([[0.25, 0.75]]), 0.1)
        log_p = torch.log_softmax(queries[:1] @ keys.T / 0.1, dim=1)[0]
        self.assertTrue(torch.allclose(soft, -(0.25 * log_p[1] + 0.75 * log_p[2])))

    def test_explicit_folds_reproduce_the_default_halves(self):
        features, targets = torch.randn(8, 4, 3, 3), torch.randn(8, 2, 3, 3)
        weights = torch.ones(8)
        default, _ = spatial_cross_fit_loss(features, targets, weights, 1.0)
        explicit, _ = spatial_cross_fit_loss(features, targets, weights, 1.0, two_view_folds(4, "cpu"))
        self.assertTrue(torch.allclose(default, explicit))

    def _mixing(self, mask: str = "concepts") -> SpatialConceptDistillation:
        # Factor 0 fills the left half of every image, factor 1 the right half; image 5 is held out.
        probabilities = torch.zeros(6, 2, 14, 14)
        probabilities[:, 0, :, :7] = 1.0
        probabilities[:, 1, :, 7:] = 1.0
        distillation = SpatialConceptDistillation(
            torch.randn(6, 2, 14, 14), 1.0, 0, 0, sample_weights=torch.tensor([1.0, 1, 1, 1, 1, 0]), ridge=1.0,
            probabilities=probabilities, mix={"fraction": 1.0, "mask": mask, "weight": 1.0, "region": True,
                                              "global": "soft"}, temperature=0.1,
        )
        distillation.set_epoch(1)
        return distillation

    def _batch(self, side: int = 28):
        views = [torch.rand(6, 3, side, side), torch.rand(6, 3, side, side)]
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0]]).repeat(6, 2, 1)
        return views, (views, torch.arange(6), boxes)

    def test_a_mixed_view_keeps_whole_concept_halves_and_stays_in_its_fold(self):
        torch.manual_seed(0)
        distillation = self._mixing()
        views, batch = self._batch()
        composed = distillation.compose([view.clone() for view in views], batch)
        self.assertTrue(torch.equal(composed[1], views[1]))
        plan = distillation.mix_plan
        self.assertNotIn(5, plan["images"].tolist() + plan["donors"].tolist())
        self.assertTrue(bool(((plan["images"] < 3) == (plan["donors"] < 3)).all()))
        self.assertFalse(bool((plan["images"] == plan["donors"]).any()))
        self.assertTrue(torch.allclose(plan["area"], torch.full_like(plan["area"], 0.5), atol=0.05))
        for row, (image, donor) in enumerate(zip(plan["images"].tolist(), plan["donors"].tolist())):
            left = bool(plan["masks"][row, 0, :, 0].mean() > 0.5)
            kept, other = (slice(0, 12), slice(16, 28)) if left else (slice(16, 28), slice(0, 12))
            self.assertTrue(torch.allclose(plan["mixed"][row][..., kept], views[1][image][..., kept]))
            self.assertTrue(torch.allclose(plan["mixed"][row][..., other], views[1][donor][..., other]))

    def test_the_box_control_keeps_a_rectangle_of_the_same_area(self):
        torch.manual_seed(0)
        distillation = self._mixing("boxes")
        views, batch = self._batch()
        distillation.compose([view.clone() for view in views], batch)
        plan = distillation.mix_plan
        self.assertIsNotNone(plan["rectangles"])
        areas = plan["rectangles"][:, 2] * plan["rectangles"][:, 3]
        self.assertTrue(torch.allclose(areas, torch.full_like(areas, 0.5), atol=0.05))

    def test_the_mixing_losses_train_the_encoder(self):
        from cospro.models.simclr import SimCLRModel

        torch.manual_seed(0)
        model = SimCLRModel("resnet18")
        distillation = self._mixing()
        distillation.hook(model)
        views, batch = self._batch(side=64)
        distillation.compose(views, batch)
        distillation.boxes = batch[2]
        embeddings = model.encoder(torch.cat(views))
        value = distillation(embeddings, batch[1], model)
        self.assertTrue(torch.isfinite(value))
        for key in ("factor_mix_fraction", "factor_mix_global_loss", "factor_mix_region_loss",
                    "factor_spatial_heldout_explained_variance"):
            self.assertIn(key, distillation.last_diagnostics)
        value.backward()
        self.assertGreater(float(model.encoder.layer4[0].conv1.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
