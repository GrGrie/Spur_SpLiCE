"""Moved to ``cospro.cli.linear_probe``. This path keeps imports and ``python -m`` commands written against it working."""

import sys

if __name__ == "__main__":
    import runpy

    runpy.run_module("cospro.cli.linear_probe", run_name="__main__", alter_sys=True)
else:
    import importlib

    sys.modules[__name__] = importlib.import_module("cospro.cli.linear_probe")
