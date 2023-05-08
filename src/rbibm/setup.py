#!/usr/bin/env python

from distutils.core import setup
import os

# Setup
setup(setup_requires=["setup.cfg"], setup_cfg=True)

# Fixed working directory!
with open("config/config.yaml", "r") as f:
    lines = f.readlines()

for i, l in enumerate(lines):
    if "data_path: " in l:
        lines[i] = l[:11] + os.getcwd() + "\n" 
        break


with open("config/config.yaml", "w") as f:
    f.writelines(lines)
