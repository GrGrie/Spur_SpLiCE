# Third-party code

This directory holds code written elsewhere and kept here so the project runs without extra
installs. The project's own code lives in `cospro/`. Local changes are listed below so a reader can
tell the original work from ours; each changed spot in the code also says "local change".

## `splice/` — SpLiCE

- Source: <https://github.com/AI4LIFE-GROUP/SpLiCE> (`splice/splice.py`, `splice/model.py`,
  `splice/admm.py`), by Usha Bhalla, Alex Oesterling, Suraj Srinivas, Flavio P. Calmon and
  Himabindu Lakkaraju: *Interpreting CLIP with Sparse Linear Concept Embeddings (SpLiCE)*,
  NeurIPS 2024.
- License: Apache License 2.0, reproduced in `splice/LICENSE`.
- Import path: `third_party.splice`. The historical path `splice` still re-exports it.

Local changes, compared with the upstream `main` branch as of 2026-09-21:

| Where | Change | Why |
|---|---|---|
| `splice.py`, `SUPPORTED_MODELS` / `SUPPORTED_VOCAB` | Only OpenCLIP ViT-B-32; vocabularies `laion` and `openimages_v7` (upstream: OpenAI CLIP models, `laion_bigrams`, `mscoco`) | The project uses one backbone and these two vocabularies |
| `splice.py`, `load(pretrained=...)` | The OpenCLIP checkpoint is a parameter and anything other than `laion2b_s34b_b79k` is refused | The bundled image mean is calibrated for that checkpoint only |
| `splice.py`, `LOCAL_DATA_ROOT`, `_resource_source`, `_download` | Vocabularies and image means are read from the repository's `data/` before any download, and local paths are accepted | Offline cluster nodes and reproducible inputs |
| `splice.py`, `_vocabulary_path`, `_clean_openimages_class_names`, `download_openimages_vocabulary` | Open Images V7 class names as a vocabulary, built from the official class-description CSV | A second, curated vocabulary |
| `splice.py`, `_select_vocabulary_lines` | A size keeps the tail of LAION (frequency-ordered) and the head of Open Images (official order) | Open Images has no frequency ranking |
| `splice.py`, embedding cache name | Includes the model and the OpenCLIP checkpoint | Two checkpoints never share cached embeddings |
| `splice.py`, `embed_concepts`, `_cached_concept_embeddings`, `load(words=..., dictionary_id=...)` | The concept-embedding loop is factored out unchanged and also accepts an explicit word list under a content-derived cache name | File concept dictionaries (`cospro/pipeline/dictionary.py`) |
| `model.py`, `SPLICE.__init__` | `text_mean` compared with `None` instead of by truth value; an unsupported solver raises instead of returning the error; ADMM runs on the model's device instead of a hard-coded `"cuda"` | Bug fixes: a tensor mean raised, an unknown solver passed silently, CPU runs failed |
| `__init__.py` | `from .splice import *` (upstream: empty) | `import splice` exposes `load` as the upstream README shows |

`admm.py` is unchanged.

## `wilds_compat.py` — a slice of WILDS

- Source: <https://github.com/p-lambda/wilds> (`WILDSDataset`, `WILDSSubset`, `CombinatorialGrouper` and the
  standard loaders), by the WILDS team.
- License: MIT, reproduced in `LICENSE.wilds`.
- Import path: `third_party.wilds_compat`.

Local changes: only the classes and loader helpers the dataset adapters use are kept, so the
project has no WILDS dependency; `get_ssl_train_loader` is added for the two-crop SimCLR loader.
The project's dataset contract builds on it in `cospro/data/base.py`.
