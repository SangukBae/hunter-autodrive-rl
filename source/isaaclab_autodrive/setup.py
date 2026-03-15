# Copyright (c) 2024, Hunter Autodrive Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'isaaclab_autodrive' python package."""

import os
from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "numpy<2",
    "torch>=2.4",
    "scipy",
]

setup(
    name="isaaclab_autodrive",
    version="0.1.0",
    description="Hunter SE 자율주행 핵심 모듈 — 에셋, 지형, 유틸리티",
    keywords=["robotics", "autonomous-driving", "hunter", "isaaclab"],
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    zip_safe=False,
)
