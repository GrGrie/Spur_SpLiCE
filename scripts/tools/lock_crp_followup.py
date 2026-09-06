"""Lock existing frozen CRP artifacts into a follow-up control configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from splice.graph_io import graph_fingerprint


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def lock_config(template_path: Path, source_output: Path, output_path: Path) -> Path:
    config = _read_json(template_path)
    cache_identity = _read_json(source_output / "cache_identity.json")
    graph_identity = _read_json(source_output / "graph_identity.json")
    cache_path = Path(cache_identity["path"])
    if not cache_path.is_file():
        raise FileNotFoundError(f"Locked cache is missing: {cache_path}")

    arms = tuple(config["arms"])
    required = set()
    if any(arm in {"crp_sampler_only", "splice_crp_kl"} for arm in arms):
        required.add("crp")
    if any(arm.startswith("raw_clip_") for arm in arms):
        required.add("raw_clip")
    paths = {
        name: source_output / "graphs" / f"{name}_graph.json"
        for name in required
    }
    fingerprints = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Locked graph is missing: {path}")
        actual = graph_fingerprint(path)
        expected = graph_identity.get(name)
        if actual != expected:
            raise ValueError(f"Source graph identity mismatch for {name}: {path}")
        fingerprints[name] = actual

    resolved = dict(config)
    resolved["cache"] = str(cache_path)
    resolved["prebuilt_graph_paths"] = {name: str(path) for name, path in paths.items()}
    resolved["prebuilt_graph_fingerprints"] = fingerprints
    resolved["locked_artifact_fingerprints"] = {
        "cache": graph_fingerprint(cache_path),
        "graphs": fingerprints,
        "source_output": str(source_output.resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    print(f"Wrote locked follow-up configuration: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lock_config(args.template, args.source_output, args.output)


if __name__ == "__main__":
    main()
