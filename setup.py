# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import platform
import os

with open("README.rst", encoding='utf-8') as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()
if platform.system == 'Linux' or platform.system == 'Darwin':
    if os.system("grep avx512 /proc/cpuinfo") == 0:
        for i, pkg in enumerate(required):
            pkg = pkg.split("==")
            if pkg[0] == "tensorflow":
                pkg[0] = "intel-tensorflow-avx512"
            required[i] = "==".join(pkg)

setup(
    name="deepinterpolation",
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
