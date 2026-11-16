"""Concept learning curves and the early concept swap, as a SimCLR training method.

Training reads no label. While SimCLR trains, the audit measures after chosen epochs how far occluding
each concept unit's regions moves the encoder's embedding of the training images (``embedding_audit``).
At the end of the early window the units that led these learning curves are flagged, and from the next
epoch on the flagged regions of a share of the views are exchanged with those of other images, so the
two views of an image agree only on what lies outside the flagged regions. Everything else is the
unchanged SimCLR run: loader order, objective, probe and storage. docs/PROJECT_PLAN.md specifies it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cospro.curves.core import (
    acquired,
    build_units,
    concept_shares,
    describe_units,
    detection_auroc,
    embedding_audit,
    flag_units,
    flagged_area,
    hard_regions,
    load_unit_maps,
    onset_scores,
    unit_presence,
    unit_typing,
)
from cospro.curves.swap import flagged_masks, swap_flagged
from cospro.data.registry import dataset_class
from cospro.methods.base import LoaderContext, TrainingMethod, register_method
from cospro.methods.concept_factors import IndexedViewBoxDataset

#: Seed of the control that gives every image another image's regions.
SHUFFLE_SEED = 0
#: Offset of the swap's own random stream from the run seed.
SWAP_SEED_OFFSET = 3_000_000


class AuditImages(Dataset):
    """The training images without augmentation, with their row: (image, row). No annotation is read."""

    def __init__(self, source, indices, image_size: int, mean, std) -> None:
        self.source = source
        self.indices = [int(index) for index in indices]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean, std),
        ])

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, row: int):
        return self.transform(self.source.get_input(self.indices[row])), row


@register_method
class ConceptCurves(TrainingMethod):
    """SimCLR with the concept audit and, once units are flagged, the swap of their regions."""

    name = "concept_curves"
    modes = ("concept_curves",)
    needs_sample_indices = True

    def __init__(self, *, concept_maps: str, seed: int = 0, unit_count: int = 12, early_epochs: int = 50,
                 audit_every: int = 5, late_audit_every: int = 50, total_epochs: int = 500,
                 statistic: str = "shift", flag_rule: str = "top", tau: float = 0.5, max_flagged_area: float = 0.75,
                 flag_units_list: str = "", flag_seed: int = 0, swap_probability: float = 0.5,
                 swap_warmup_epochs: int = 10, swap_views: str = "both", shuffle_maps: bool = False,
                 image_size: int | None = None) -> None:
        self.concept_maps = concept_maps
        self.seed = int(seed)
        self.unit_count = int(unit_count)
        self.early_epochs = int(early_epochs)
        self.audit_every = max(int(audit_every), 1)
        self.late_audit_every = int(late_audit_every)
        self.total_epochs = int(total_epochs)
        self.statistic = statistic
        self.flag_rule = flag_rule
        self.tau = float(tau)
        self.max_flagged_area = float(max_flagged_area)
        self.explicit_units = [int(unit) for unit in str(flag_units_list).split(",") if unit.strip()]
        self.flag_seed = int(flag_seed)
        self.swap_probability = float(swap_probability)
        self.swap_warmup_epochs = int(swap_warmup_epochs)
        self.swap_views = swap_views
        self.shuffle_maps = bool(shuffle_maps)
        self.image_size = image_size
        self.epoch = 0
        self.regions: torch.Tensor | None = None
        self.swap_regions: torch.Tensor | None = None
        self.units: list[dict] = []
        self.audited: list[int] = []
        self.source_indices: list[int] = []
        self.audit_loader = None
        self.flagged: list[int] | None = None
        self.flags: dict | None = None
        self.curves: dict[str, Any] = {}
        self.generator = torch.Generator().manual_seed(self.seed + SWAP_SEED_OFFSET)
        self.last_diagnostics: dict[str, float] = {}

    @classmethod
    def from_config(cls, config, **resolved) -> "ConceptCurves":
        options = config.concept_curves
        return cls(
            concept_maps=options.curve_concept_maps,
            seed=config.runtime.seed,
            unit_count=options.curve_unit_count,
            early_epochs=options.curve_early_epochs,
            audit_every=options.curve_audit_every,
            late_audit_every=options.curve_late_audit_every,
            total_epochs=config.optimization.epochs,
            statistic=options.curve_statistic,
            flag_rule=options.curve_flag_rule,
            tau=options.curve_tau,
            max_flagged_area=options.curve_max_flagged_area,
            flag_units_list=options.curve_flag_units,
            flag_seed=options.curve_flag_seed,
            swap_probability=options.curve_swap_probability,
            swap_warmup_epochs=options.curve_swap_warmup_epochs,
            swap_views=options.curve_swap_views,
            shuffle_maps=options.curve_shuffle_maps,
        )

    # ------------------------------------------------------------------ loader

    def wrap_loader(self, loader, context: LoaderContext):
        subset = loader.dataset
        self.source_indices = [int(index) for index in subset.indices]
        maps, factor_names = load_unit_maps(self.concept_maps, context.dataset, self.source_indices)
        unit_of_factor = build_units(maps, self.unit_count, seed=0)
        self.regions = hard_regions(maps, unit_of_factor)
        self.units = describe_units(maps, unit_of_factor, self.regions, factor_names)
        self.audited = [entry["unit"] for entry in self.units if entry["audited"]]
        self.swap_regions = self.regions
        if self.shuffle_maps:
            order = torch.randperm(len(self.regions), generator=torch.Generator().manual_seed(SHUFFLE_SEED))
            self.swap_regions = self.regions[order]
        self.curves = {
            "dataset": context.dataset, "seed": self.seed, "concept_maps": str(self.concept_maps),
            "statistic": self.statistic, "unit_count": len(self.units), "early_epochs": self.early_epochs,
            "unit_of_factor": unit_of_factor.tolist(), "units": self.units,
            "epochs": [], "shift": [], "margin": [], "share": [], "onset": None, "evaluation_only": {},
        }
        for entry in self.units:
            if entry["audited"]:
                print(f"[INFO] Concept unit {entry['unit']} (presence {entry['presence']:.2f}, area "
                      f"{entry['area']:.2f}): {entry['name']}", flush=True)
        adapter = dataset_class(context.dataset)
        size = self.image_size or adapter.Config().image_size
        self.audit_loader = DataLoader(
            AuditImages(subset.dataset, self.source_indices, size, adapter.mean, adapter.std),
            batch_size=context.batch_size, shuffle=False, num_workers=context.num_workers, pin_memory=True,
        )
        indexed = IndexedViewBoxDataset(subset)
        # The sampler and generator of the SimCLR loader are kept, so batches follow the SimCLR order.
        return DataLoader(
            indexed, batch_size=loader.batch_size, sampler=loader.sampler, drop_last=loader.drop_last,
            num_workers=context.num_workers, pin_memory=True, collate_fn=indexed.collate,
            worker_init_fn=context.worker_init_fn if context.num_workers > 0 else None, generator=loader.generator,
        )

    # ------------------------------------------------------------------- audit

    def audit_due(self, epoch: int) -> bool:
        """Epoch 0, every ``audit_every`` epochs of the early window, its last epoch, then the late interval."""

        if epoch in (0, self.early_epochs, self.total_epochs):
            return True
        if epoch < self.early_epochs:
            return epoch % self.audit_every == 0
        return self.late_audit_every > 0 and epoch % self.late_audit_every == 0

    def audit(self, model, device, amp: bool, epoch: int) -> dict[str, float]:
        """Measure the reliance on every audited unit after ``epoch`` and extend the curves."""

        result = embedding_audit(model, self.audit_loader, self.regions, self.audited, device, amp=amp)
        curves = self.curves
        curves["epochs"].append(int(epoch))
        curves["shift"].append(result["shift"])
        curves["margin"].append(result["margin"])
        values = curves[self.statistic]
        curves["share"].append(concept_shares(acquired(values[-1], values[0]), self.audited))
        if any(1 <= done <= self.early_epochs for done in curves["epochs"]):
            curves["onset"] = onset_scores(curves["epochs"], curves["share"], self.early_epochs)
        share = curves["share"][-1]
        leader = max(self.audited, key=lambda unit: share[unit])
        print(f"[INFO] Concept audit at epoch {epoch}: leading unit {leader} ({self.units[leader]['name']}) "
              f"with share {share[leader]:.2f}", flush=True)
        return {"Concept audit leading share": share[leader]}

    def score_audit(self, labels, attributes) -> None:
        """Type the units against hidden labels and score the onset ranking. Evaluation only."""

        typing = unit_typing(unit_presence(self.regions), torch.as_tensor(labels), torch.as_tensor(attributes))
        evaluation = self.curves["evaluation_only"]
        evaluation["typing"] = typing
        if self.curves["onset"] is not None:
            score = detection_auroc(self.curves["onset"], self.units, typing)
            evaluation["detection_auroc"] = None if score != score else score

    # ---------------------------------------------------------------- flagging

    def decide_flags(self) -> dict | None:
        """Flag units from the curves of the early window; called once, after the audit of its last epoch."""

        if self.flag_rule == "none" or self.flags is not None:
            return self.flags
        typing = self.curves["evaluation_only"].get("typing")
        if self.flag_rule == "oracle":
            if typing is None:
                raise ValueError("The oracle rule needs the hidden-label typing of the units.")
            flagged = [entry["unit"] for entry in typing if entry["type"] == "attribute" and entry["unit"] in self.audited]
            if not flagged:
                raise ValueError("The oracle rule found no attribute unit.")
        else:
            try:
                flagged = flag_units(self.curves["onset"], self.units, rule=self.flag_rule, tau=self.tau,
                                     max_area=self.max_flagged_area, seed=self.flag_seed, explicit=self.explicit_units)
            except ValueError as error:
                # Without a positive acquired reliance there is nothing to flag; the run continues as SimCLR.
                print(f"[WARNING] No concept unit flagged: {error}", flush=True)
                self.flags = {"rule": self.flag_rule, "tau": self.tau, "flagged": [], "names": [], "types": None,
                              "decided_after_epoch": self.early_epochs, "reason": str(error)}
                return self.flags
        self.restore_flags({
            "rule": self.flag_rule, "tau": self.tau, "flagged": flagged, "decided_after_epoch": self.early_epochs,
            "names": [self.units[unit]["name"] for unit in flagged],
            "types": None if typing is None else [typing[unit]["type"] for unit in flagged],
            "mean_flagged_area": round(flagged_area(flagged, self.units), 4), "shuffled_maps": self.shuffle_maps,
        })
        print(f"[INFO] Flagged concept units ({self.flag_rule}): {self.flags['names']}", flush=True)
        return self.flags

    def restore_flags(self, flags: dict) -> None:
        self.flags = dict(flags)
        self.flagged = [int(unit) for unit in flags["flagged"]] or None

    # -------------------------------------------------------------------- swap

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    @property
    def scheduled_probability(self) -> float:
        """Zero through the early window, then a linear ramp to the swap probability."""

        if self.flagged is None or self.epoch <= self.early_epochs:
            return 0.0
        if self.swap_warmup_epochs <= 0:
            return self.swap_probability
        return self.swap_probability * min(1.0, (self.epoch - self.early_epochs) / self.swap_warmup_epochs)

    def compose_views(self, views, batch):
        probability = self.scheduled_probability
        self.last_diagnostics = {"curve_swap_probability": probability, "curve_swapped_fraction": 0.0}
        if probability <= 0 or len(batch) < 3:
            return views
        rows = torch.as_tensor(batch[1], dtype=torch.long).view(-1)
        regions = self.swap_regions[rows].to(views[0].device)
        boxes = batch[2].to(views[0].device)
        swapped_total = 0.0
        targets = (0, 1) if self.swap_views == "both" else (1,)
        with torch.no_grad():
            for view in targets:
                masks = flagged_masks(regions, self.flagged, boxes[:, view], tuple(views[view].shape[-2:]))
                views[view], swapped = swap_flagged(views[view], masks, probability, generator=self.generator)
                swapped_total += float(swapped.float().mean())
        self.last_diagnostics["curve_swapped_fraction"] = swapped_total / len(targets)
        return views

    # ------------------------------------------------------------ bookkeeping

    def diagnostics(self) -> Mapping[str, float]:
        return self.last_diagnostics

    def provenance(self) -> dict[str, Any]:
        return {"concept_curve_units": [entry["name"] for entry in self.units if entry["audited"]]}

    def input_artifacts(self) -> list[Path]:
        # The maps are a large binary on scratch; the provenance and concept_curves.json name them.
        return []
