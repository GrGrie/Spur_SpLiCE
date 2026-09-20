# Adding a dataset

A dataset in this project pairs a target label `y` with a spurious attribute `a` and splits its
samples into train, validation and test. Everything else, including the transforms, the group
report, the four loader roles and the model the image size allows, comes from
`experiments/spurious_eval/datasets/base.py`. Adding a dataset is therefore an adapter plus a
handful of attributes, followed by the pipeline stages that turn it into a teacher graph.

## 1. The adapter

Create `experiments/spurious_eval/datasets/<name>.py`:

```python
@dataclass(frozen=True)
class MyDatasetConfig(DatasetConfig):
    """Only the fields this dataset adds. Everything else is inherited."""


@register_dataset
class MyDataset(SpuriousDataset):
    name = "my_dataset"          # canonical spelling; sample IDs embed it
    aliases = ("my-dataset",)    # spellings accepted at the command line
    num_classes = 2
    image_size = 224             # below 224 the 224-pixel ResNet stem is refused
    Config = MyDatasetConfig

    def __init__(self, root_dir: str = "./datasets", split_scheme: str = "official") -> None:
        ...   # read the metadata, then call super().__init__(root_dir, split_scheme)

    def get_input(self, idx: int):
        ...   # return one PIL image
```

`__init__` reads the metadata and sets the WILDS-style attributes the base class expects:

| Attribute | Content |
|---|---|
| `_y_array` | target label per sample, as a `LongTensor` |
| `_metadata_array` | one row per sample: the spurious attribute first, the target second |
| `_metadata_fields`, `_metadata_map` | field names and the display names of their values |
| `_split_array` | 0 for train, 1 for validation, 2 for test |
| `_eval_grouper` | `CombinatorialGrouper` over the two metadata fields |
| `_data_dir` | the resolved dataset root, from `resolve_dataset_root` |

Override `from_config` when the dataset is built from more than a root directory; `spur_cifar10`
draws its spurious correlation at construction and shows the pattern. Override
`spurious_metadata_index` or `target_metadata_index` only when the metadata columns are in another
order. Keep `mean` and `std` at the ImageNet defaults unless the images call for their own.

Import the module in `experiments/spurious_eval/datasets/registry.py`. That import is the
registration: no other file in the project carries a list of dataset names. The command-line
choices, the class count, the default model and the diagnostics label reader all follow from it.

Nothing else builds loaders by hand. `build_loader(MyDataset, role, config, batch_size)` serves the
four roles:

| Role | Split | Transform | Order |
|---|---|---|---|
| `ssl` | `train` | two augmented crops | shuffled |
| `rank` | `train` | evaluation transform | in order |
| `probe_train` | `config.train_split` | augmented | shuffled |
| `probe_eval` | `config.eval_split` | evaluation transform | in order |

## 2. The SpLiCE cache

The cache freezes the CLIP embeddings and sparse concept codes of the train split. The cluster
builds it, since it needs a GPU:

```bash
DATA_FOLDER=/path/to/datasets sbatch scripts/cache_splice_dataset.sh my_dataset
```

The launcher prints where the cache lands. Sample IDs in the cache are `my_dataset:<metadata row>`,
which is how every later stage aligns groups, graphs and labels with the metadata.

## 3. Concept groups, teacher graph and diagnostics

```bash
sbatch scripts/run_cospro_pipeline.sh --dataset my_dataset          # cache, groups, graph, training
sbatch scripts/run_cospro_diagnostics.sbatch                        # DATASET=my_dataset
```

The diagnostics job writes `outputs/reports/cospro_diagnostics/my_dataset/diagnostics.json`; render
it anywhere:

```bash
python -m cospro.diagnostics dashboard outputs/reports/cospro_diagnostics/my_dataset/diagnostics.json \
    --data-folder /path/to/datasets --output tmp/diagnostics.html
```

The dashboard reads the labels through the adapter, so the class and attribute names in the report
are the ones `_metadata_map` declares.

## 4. The manifest

A study is a matrix of seeds by arms in `experiments/manifests/<study>.yaml`, whose keys are
`spur_splice.py` options without their leading dashes:

```yaml
name: my_dataset_cospro
seeds: [1, 2, 3, 4]
flags: [use_wandb, keep_checkpoints]
common:
  dataset: my_dataset
  data_folder: ${DATA_FOLDER}
  epochs: 500
arms:
  simclr: {args: {splice_mode: none}}
  cospro: {args: {splice_mode: cospro_relational, cospro_teacher_graph: ...}}
```

Leave `model` out and the dataset's `default_model()` applies.

## 5. What to check

```bash
python -m pytest tests/test_dataset_adapters.py -q
python -m pytest tests/test_golden_training.py -q     # unchanged numbers for existing datasets
```

`tests/test_dataset_adapters.py` covers the registry and the loader roles for every registered
dataset, so a new adapter is exercised by it as soon as it is imported in `registry.py`.
