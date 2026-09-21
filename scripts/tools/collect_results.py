"""Moved to ``cospro.cli.collect_results``. This path keeps imports and ``python -m`` commands written against it working."""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("cospro.cli.collect_results", run_name="__main__", alter_sys=True)
else:
    import importlib

    sys.modules[__name__] = importlib.import_module("cospro.cli.collect_results")
