#!/bin/sh
# get latest version of 'build'
pip install --upgrade build
# actually build the library
python -m build

