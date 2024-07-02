#!/bin/sh
# get latest version of 'build'
python3 -m pip install --upgrade build
# actually build the library
python3 -m build

