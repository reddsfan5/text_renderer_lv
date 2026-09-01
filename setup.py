#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="text_renderer",
    version="0.1.0",
    author="oh-my-ocr",
    url="https://github.com/oh-my-ocr/text_renderer",
    description="",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["PyYAML>=5.1"],
    entry_points={"console_scripts": ["text-renderer=text_renderer.cli:main"]},
)
