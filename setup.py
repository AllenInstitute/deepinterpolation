# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import platform
import os

with open("README.rst", encoding='utf-8') as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

if platform.system == 'Linux':
    if os.system("grep avx512 /proc/cpuinfo") == 0:
        with open("requirements-avx512.txt", "r") as f:
            required = f.read().splitlines()
    else:
        with open("requirements.txt", "r") as f:
            required = f.read().splitlines()
else:
    with open("requirements.txt", "r") as f:
        required = f.read().splitlines()

setup(
    name="deepinterpolation",
    version="0.1.5",
    description="Implemenent DeepInterpolation to denoise data by removing \
independent noise",
    long_description_content_type='text/x-rst',
    long_description=readme,
    author="Jerome Lecoq",
    author_email="jeromel@alleninstitute.org",
    url="https://github.com/AllenInstitute/deepinterpolation",
    license=license,
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=required,
)
