import os
from setuptools import setup, find_packages


def _read_requirements(req_path: str):
    reqs = []
    try:
        with open(req_path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                reqs.append(line)
    except FileNotFoundError:
        return []
    return reqs


requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

setup(
    name="splice",
    version="1.0",
    description="",
    author="Alex Oesterling, Usha Bhalla",
    author_email="aoesterling@g.harvard.edu, usha_bhalla@g.harvard.edu",
    py_modules=["splice"],
    packages=find_packages(exclude=["experiments*", "data*"]),
    install_requires=_read_requirements(requirements_path),
)