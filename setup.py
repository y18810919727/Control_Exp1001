#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Control_Exp1001",
    version="0.0.1",
    author="Zhaolin Yuan",
    author_email="b20170324@xs.ustb.edu.cn",
    description="A highly integrated platform for controlling experiments in 1001 ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)