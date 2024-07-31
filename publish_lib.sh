#!/bin/sh
# build the library
./build_lib.sh
# get latest version of 'twine'
python3 -m pip install --upgrade twine
if [ $# -eq 0 ]
then
  echo "Publishing to test.pypi. If you want to publish for real, use the argument 'notest'"
  # publish to test.pypi
  twine upload -r testpypi dist/*
elif [ $1 = notest ]
then
  echo "Publishing to pypi."
  twine upload dist/*
else
  echo "Unexpected argument $1"
  echo "For publishing to pypu use 'notest', or provide no argument to publish to test.pypi."
fi

