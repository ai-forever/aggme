import pathlib

from setuptools import find_packages, setup

PREFIX = "aggme"
here = pathlib.Path(__file__).parent.resolve()
version = (here / "VERSION").read_text(encoding="utf-8")
long_description = (here / "README.md").read_text(encoding="utf-8")
requirements = (here / "requirements.txt").read_text(encoding="utf-8")


setup(
    name="aggme",
    version=version,
    description="Markup aggregation utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SberDevices",
    python_requires=">=3.10.0",
    install_requires=[requirements],
    package_dir={PREFIX: PREFIX},
    packages=[
        f"{PREFIX}",
        *(f"{PREFIX}.{package}" for package in find_packages(f"{PREFIX}")),
    ],
)
