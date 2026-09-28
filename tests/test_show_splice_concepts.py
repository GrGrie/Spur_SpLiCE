"""The SpLiCE concept page: concept ranking and rendering, without CLIP."""

from __future__ import annotations

import unittest

import torch
from PIL import Image

from cospro.cli.show_splice_concepts import Entry, parse_args, render_page, top_concepts


class ConceptPageTests(unittest.TestCase):
    def test_top_concepts_are_the_strongest_positive_weights(self):
        code = torch.tensor([0.0, 0.5, 0.1, 0.9, 0.0])
        concepts = top_concepts(code, ["a", "b", "c", "d", "e"], 4)
        self.assertEqual(concepts, [("d", 0.9), ("b", 0.5), ("c", 0.1)])

    def test_page_holds_every_image_and_escapes_concept_names(self):
        entry = Entry(title="cat.jpg", image=Image.new("RGB", (32, 32)), concepts=[("<Cat>", 0.4), ("Sofa", 0.2)],
                      active=2, reconstruction=0.5, details=["y = cat"])
        page = render_page([entry, entry], "2 images")
        self.assertEqual(page.count('<section class="card">'), 2)
        self.assertIn("&lt;Cat&gt;", page)
        self.assertIn("data:image/jpeg;base64,", page)
        self.assertIn("y = cat", page)

    def test_random_images_need_a_dataset(self):
        with self.assertRaises(SystemExit):
            parse_args(["--random", "3"])
        with self.assertRaises(SystemExit):
            parse_args([])


if __name__ == "__main__":
    unittest.main()
