#!/usr/bin/env python3
"""
Mostly taken from https://github.com/rochacbruno/python-project-template/blob/main/setup.py
"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-", "git+"))]


setup(
    name="poptimizer",
    version=read("src", "poptimizer", "VERSION"),
    description="Repository for gauging language model performance using likelihood.",
    url="https://github.com/lyutyuh/poptimizer",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=read_requirements("requirements.txt"),
)
