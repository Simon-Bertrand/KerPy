#!/bin/bash
rm -rf build
rm -rf dist
python setup.py sdist bdist_wheel
twine upload --skip-existing -r testpypi dist/* -u "Simon-Bertrand" -p $1