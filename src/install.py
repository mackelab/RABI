#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import sys
import os

here = os.path.abspath(os.path.dirname(__file__))
sep = os.sep

def install_v1():
    os.system(f"pip install -e '{here}{sep}v1{sep}rbi'")
    os.system(f"pip install -e '{here}{sep}v1{sep}rbibm'")

def install_v2():
    os.system("pip install -e v1/rbi")
    os.system("pip install -e v1/rbibm")

if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        version = args[-1]
    else:
        version = "v1"
    print(f"{here}{sep}v1{sep}rbi")
    if version == "v1":
        install_v1()
    elif version == "v2":
        install_v2()

