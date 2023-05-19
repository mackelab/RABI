#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import sys
import os

here = os.path.abspath(os.path.dirname(__file__))
sep = os.sep


def install():
    os.system(f"pip install -e {here}{sep}rbi --user")
    os.system(f"pip install -e {here}{sep}rbibm --user")


def install_r():
    os.system(f"pip install -r {here}{sep}requirements.txt --user")


if __name__ == "__main__":
    args = sys.argv
    if len(args) > 1:
        with_requirements = args[-1] == "-r"
    else:
        with_requirements = False

    install()

    if with_requirements:
        install_r()
