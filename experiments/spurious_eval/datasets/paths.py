from __future__ import annotations

import os
from pathlib import Path


def resolve_dataset_root(root_dir: str | Path, dataset_name: str, marker_paths: list[str]) -> Path:
    root_path = Path(root_dir).expanduser()
    normalized_name = dataset_name.lower()

    env_candidates = [
        os.environ.get("DATASET_ROOT"),
        os.environ.get("DATASETS_ROOT"),
        os.environ.get("SPURSSL_DATA_ROOT"),
    ]
    search_roots = [root_path]
    for candidate in env_candidates:
        if candidate:
            search_roots.append(Path(candidate).expanduser())

    parent_candidates = [
        root_path.parent,
        Path.home() / "Datasets",
        Path.home() / "datasets",
    ]
    for candidate in parent_candidates:
        if candidate not in search_roots:
            search_roots.append(candidate)

    candidate_dirs: list[Path] = []
    seen: set[Path] = set()
    dataset_aliases = [dataset_name, normalized_name, normalized_name.replace("_", ""), normalized_name.replace("_", "-")]

    for search_root in search_roots:
        for candidate in [search_root, *(search_root / alias for alias in dataset_aliases)]:
            resolved = candidate.resolve(strict=False)
            if resolved not in seen:
                seen.add(resolved)
                candidate_dirs.append(candidate)

    for candidate in candidate_dirs:
        if all((candidate / marker).exists() for marker in marker_paths):
            return candidate

    searched = ", ".join(str(path) for path in candidate_dirs)
    marker_text = ", ".join(marker_paths)
    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}' with markers [{marker_text}]. Searched: {searched}"
    )
